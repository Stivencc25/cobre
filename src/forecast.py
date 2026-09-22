"""F3 — Forecast probabilístico del FLUJO NETO diario (no del nivel del balance).

Escalera de modelos (SPEC F3): (1) naive estacional semanal = baseline obligatorio,
(2) ETS aditivo con estacionalidad 7, (3) SARIMAX con dummies de día de semana y fin de mes,
(4) Prophet (tendencia con changepoints + estacionalidad semanal) como reto adicional al SARIMAX.
Se incluye además ``empirical`` (pronóstico puntual = 0 y distribución = bootstrap i.i.d. de los
flujos históricos por tipo de día) como chequeo de cordura: si nadie le gana, la media del flujo
neto no tiene señal explotable y todo el valor está en la distribución, no en el punto.
Ojo: en ``empirical`` los "residuos" son los propios flujos, así que la distribución hereda la
deriva histórica (los flujos medios son negativos); el punto = 0 es solo la métrica MAE.

Distribución predictiva: bootstrap de residuos in-sample por tipo de día (lab / fin de semana),
independiente entre horizontes y entre cuentas (la correlación cruzada observada es ~0).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from cmdstanpy.utils import disable_logging as _disable_cmdstanpy_logging
from prophet import Prophet
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config as cfg

MODEL_ORDER = ("empirical", "seasonal_naive", "ets", "sarimax", "prophet")     # de más simple a más complejo
BASELINE = "seasonal_naive"


def net_flow_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Matriz fecha × cuenta del flujo neto (columnas en el orden de cuentas del panel)."""
    return panel.pivot(index="date", columns="account_id", values="net_flow").sort_index()


def calendar_exog(index: pd.DatetimeIndex) -> np.ndarray:
    """Dummies de día de semana (lunes = base) + indicador de fin de mes."""
    dow = np.asarray(index.dayofweek)
    cols = [(dow == d).astype(float) for d in range(1, 7)]
    cols.append(np.asarray(index.is_month_end).astype(float))
    return np.column_stack(cols)


def day_type(index) -> np.ndarray:
    """0 = día hábil, 1 = fin de semana."""
    return (np.asarray(pd.DatetimeIndex(index).dayofweek) >= 5).astype(int)


# ----------------------------------------------------------------------- modelos
def _fit_empirical(y, index, H):
    return np.zeros(H), y.copy(), 0


def _fit_seasonal_naive(y, index, H):
    p = cfg.SEASONAL_PERIOD
    mean = np.array([y[-p + (k % p)] for k in range(H)])
    resid = np.full_like(y, np.nan)
    resid[p:] = y[p:] - y[:-p]
    return mean, resid, 0


def _fit_ets(y, index, H):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ETSModel(y, error="add", trend=None, seasonal="add", seasonal_periods=cfg.SEASONAL_PERIOD).fit(disp=False)
    return np.asarray(res.forecast(H)), np.asarray(res.resid), len(caught)


def _fit_sarimax(y, index, H):
    future = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=H)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = SARIMAX(y, exog=calendar_exog(index), order=(1, 0, 0), trend="c").fit(disp=False)
        mean = np.asarray(res.forecast(H, exog=calendar_exog(future)))
    return mean, np.asarray(res.resid), len(caught)


def _fit_prophet(y, index, H):
    """Tendencia (changepoints automáticos) + estacionalidad semanal; sin anual/diaria (270 días, frecuencia diaria).

    ``uncertainty_samples=0``: Prophet no calcula su propia banda, porque la distribución predictiva de todo
    el módulo sale del bootstrap de residuos in-sample (misma mecánica que ETS/SARIMAX), no de cada modelo.
    """
    df = pd.DataFrame({"ds": pd.DatetimeIndex(index), "y": y})
    future = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=H)
    with warnings.catch_warnings(record=True) as caught, _disable_cmdstanpy_logging():
        warnings.simplefilter("always")
        model = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                        uncertainty_samples=0).fit(df)
        mean = model.predict(pd.DataFrame({"ds": future}))["yhat"].to_numpy()
        resid = (df["y"] - model.predict(df)["yhat"]).to_numpy()
    return mean, resid, len(caught)


FITTERS = {"empirical": _fit_empirical, "seasonal_naive": _fit_seasonal_naive, "ets": _fit_ets, "sarimax": _fit_sarimax,
           "prophet": _fit_prophet}


# ------------------------------------------------------ distribución predictiva
@dataclass
class OriginForecast:
    """Forecast de todas las cuentas desde un origen: media por día y reservas de residuos."""

    origin: pd.Timestamp
    dates: pd.DatetimeIndex
    accounts: list[str]
    mean: np.ndarray                      # (H, A)
    pools: list[tuple[np.ndarray, np.ndarray]]   # por cuenta: (residuos hábil, residuos fin de semana)
    scale: np.ndarray                     # (A,) desv. estándar de los flujos de entrenamiento
    n_warnings: int = 0
    _dtype: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._dtype = day_type(self.dates)

    def draw(self, n: int, rng: np.random.Generator, bias_sd: float = 0.0) -> np.ndarray:
        """(n, H, A) trayectorias de flujo neto = media + residuo bootstrap; ``bias_sd`` degrada la media."""
        H, A = self.mean.shape
        out = np.empty((n, H, A))
        for a in range(A):
            for typ in (0, 1):
                cols = np.where(self._dtype == typ)[0]
                pool = self.pools[a][typ]
                out[:, cols, a] = self.mean[cols, a] + rng.choice(pool, size=(n, len(cols)))
        if bias_sd:
            out += bias_sd * self.scale[None, None, :]
        return out

    def quantiles(self, n: int, rng: np.random.Generator, qs=cfg.QUANTILES):
        """Cuantiles por día y acumulados: ``(daily (H,A,Q), cum (H,A,Q))``."""
        paths = self.draw(n, rng)
        daily = np.quantile(paths, qs, axis=0).transpose(1, 2, 0)
        cum = np.quantile(np.cumsum(paths, axis=1), qs, axis=0).transpose(1, 2, 0)
        return daily, cum


def fit_origin(flows: pd.DataFrame, t: int, model: str, H: int = cfg.DECISION_HORIZON) -> OriginForecast:
    """Ajusta ``model`` con datos hasta la posición ``t`` (inclusive) y pronostica ``H`` días."""
    fitter = FITTERS[model]
    train = flows.iloc[: t + 1]
    index = train.index
    p = cfg.SEASONAL_PERIOD
    future = pd.date_range(index[-1] + pd.Timedelta(days=1), periods=H)
    means, pools, warns = [], [], 0
    resid_dt = day_type(index)
    for acc in flows.columns:
        y = train[acc].to_numpy(float)
        mean, resid, w = fitter(y, index, H)
        warns += w
        ok = np.isfinite(resid)
        ok[:p] = False                                     # se descartan los residuos de arranque
        pools.append((resid[ok & (resid_dt == 0)], resid[ok & (resid_dt == 1)]))
        means.append(mean)
    return OriginForecast(
        origin=index[-1], dates=future, accounts=list(flows.columns), mean=np.column_stack(means),
        pools=pools, scale=train.std().to_numpy(float), n_warnings=warns,
    )


# --------------------------------------------------------------------- backtest
def pinball(y: np.ndarray, q: np.ndarray, taus=cfg.QUANTILES) -> np.ndarray:
    """Pinball loss; ``q`` tiene el eje de cuantiles al final. Devuelve el mismo shape que ``q``."""
    taus = np.asarray(taus)
    diff = y[..., None] - q
    return np.maximum(taus * diff, (taus - 1) * diff)


def walk_forward(flows: pd.DataFrame, model: str, H: int = cfg.DECISION_HORIZON, step: int = 1,
                 n_draws: int = cfg.N_DRAWS_EVAL, min_train: int = cfg.MIN_TRAIN_DAYS) -> dict:
    """Backtest walk-forward con ventana expansiva. Devuelve arrays alineados por origen.

    Claves: ``origins`` (posiciones), ``actual`` (O,H,A), ``mean`` (O,H,A), ``q_daily`` (O,H,A,Q),
    ``q_cum`` (O,H,A,Q), ``actual_cum`` (O,H,A), ``mean_cum`` (O,H,A), ``n_warnings``.
    """
    y = flows.to_numpy(float)
    T = len(flows)
    origins = list(range(min_train - 1, T - H, step))
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    keep = {k: [] for k in ("actual", "mean", "q_daily", "q_cum")}
    warns = 0
    for t in origins:
        f = fit_origin(flows, t, model, H)
        qd, qc = f.quantiles(n_draws, rng)
        keep["actual"].append(y[t + 1 : t + 1 + H])
        keep["mean"].append(f.mean)
        keep["q_daily"].append(qd)
        keep["q_cum"].append(qc)
        warns += f.n_warnings
    out = {k: np.array(v) for k, v in keep.items()}
    out.update(origins=np.array(origins), actual_cum=np.cumsum(out["actual"], axis=1),
               mean_cum=np.cumsum(out["mean"], axis=1), n_warnings=warns, accounts=list(flows.columns), model=model)
    return out


def score(result: dict, horizons=cfg.HORIZONS, taus=cfg.QUANTILES, interval=cfg.INTERVAL) -> pd.DataFrame:
    """MAE, pinball y cobertura p5–p95 por cuenta y horizonte (diario y acumulado)."""
    lo, hi = list(taus).index(interval[0]), list(taus).index(interval[1])
    rows = []
    for a, acc in enumerate(result["accounts"]):
        for h in horizons:
            act, mean = result["actual"][:, :h, a], result["mean"][:, :h, a]
            qd = result["q_daily"][:, :h, a, :]
            ac, mc = result["actual_cum"][:, h - 1, a], result["mean_cum"][:, h - 1, a]
            qc = result["q_cum"][:, h - 1, a, :]
            rows.append(
                {
                    "model": result["model"], "account_id": acc, "h": h,
                    "mae_daily": np.abs(act - mean).mean(),
                    "pinball_daily": pinball(act, qd, taus).mean(),
                    "cover_daily": ((act >= qd[..., lo]) & (act <= qd[..., hi])).mean(),
                    "mae_cum": np.abs(ac - mc).mean(),
                    "pinball_cum": pinball(ac, qc, taus).mean(),
                    "cover_cum": ((ac >= qc[:, lo]) & (ac <= qc[:, hi])).mean(),
                    "below_p5_cum": (ac < qc[:, lo]).mean(),
                    "above_p95_cum": (ac > qc[:, hi]).mean(),
                }
            )
    return pd.DataFrame(rows)


def compare_models(flows: pd.DataFrame, models=MODEL_ORDER, step: int = 1, **kw):
    """Corre el walk-forward de cada modelo. Devuelve ``(tabla_de_scores, resultados_crudos)``."""
    results = {m: walk_forward(flows, m, step=step, **kw) for m in models}
    return pd.concat([score(r) for r in results.values()], ignore_index=True), results


METRICS = ("mae_daily", "pinball_daily", "mae_cum", "pinball_cum")


def skill_vs_baseline(scores: pd.DataFrame) -> pd.DataFrame:
    """Métrica / métrica del baseline (<1 = mejor que el naive estacional), por modelo, cuenta y h."""
    base = scores[scores["model"] == BASELINE].set_index(["account_id", "h"])
    out = scores.copy()
    for m in METRICS:
        out[m + "_rel"] = out.apply(lambda r: r[m] / base.loc[(r["account_id"], r["h"]), m], axis=1)
    return out


def select_model(scores: pd.DataFrame, tolerance: float = cfg.MODEL_TOLERANCE) -> tuple[str, pd.DataFrame]:
    """El más simple que (i) gana al baseline y (ii) queda a <= ``tolerance`` del mejor pinball.

    Métrica de decisión: pinball acumulado relativo al baseline, promediado entre cuentas y horizontes
    (es la pérdida que importa para dimensionar buffers). Nunca por preferencia.
    """
    rel = skill_vs_baseline(scores)
    summary = rel.groupby("model")[["mae_daily_rel", "pinball_daily_rel", "mae_cum_rel", "pinball_cum_rel"]].mean()
    summary["cover_cum"] = scores.groupby("model")["cover_cum"].mean()
    summary["cover_daily"] = scores.groupby("model")["cover_daily"].mean()
    beats = summary["pinball_cum_rel"] < 1.0
    best = summary.loc[beats, "pinball_cum_rel"].min()
    ok = [m for m in MODEL_ORDER if m in summary.index and m != BASELINE and beats[m] and summary.loc[m, "pinball_cum_rel"] <= best * (1 + tolerance)]
    return ok[0], summary.loc[list(MODEL_ORDER)]
