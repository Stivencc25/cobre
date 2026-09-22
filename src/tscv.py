"""D4b — Forecast ETS del saldo y validación con ``TimeSeriesSplit`` (ventana expansiva).

Cada corte entrena con todo el pasado hasta un origen y prueba en el bloque de ``cfg.TSCV_TEST_SIZE`` días siguientes
(multi-paso desde ese origen). Modelos comparados, todos contra el saldo real del panel limpio:

* ``squad_sin_limpiar`` / ``squad_limpio``: media de los últimos 14 (filas / días), un escalar constante.
* ``ultimo_saldo``: el saldo del origen, sin cambios (referencia ingenua).
* ``ets_nivel``: ETS sobre el saldo, tendencia amortiguada, SIN estacionalidad.
* ``ets_nivel_semanal``: lo mismo con estacionalidad semanal forzada (periodo 7).
* ``ets_flujos``: ETS de ``inflow`` y ``outflow`` por separado (estacionalidad semanal solo si el test la identifica en
  los datos de ENTRENAMIENTO de ese corte); el saldo es el último saldo + (entradas - salidas) acumulados.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from . import config as cfg
from . import seasonality
from .diagnosis import squad_raw_mean

MODELS = ("squad_sin_limpiar", "squad_limpio", "ultimo_saldo", "ets_nivel", "ets_nivel_semanal", "ets_flujos")
BASE = "squad_sin_limpiar"


def _ets(y: np.ndarray, h: int, seasonal_periods: int | None = None, trend: str | None = None) -> tuple[np.ndarray, int]:
    """Ajusta un ETS aditivo y pronostica ``h`` pasos. Devuelve ``(pronóstico, n_warnings)``.

    La serie se escala por su media absoluta (los saldos en COP llegan a 1e9 y desestabilizan el optimizador);
    los warnings de convergencia se cuentan, no se silencian.
    """
    scale = float(np.mean(np.abs(y))) or 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ETSModel(
            y / scale, error="add", trend=trend, damped_trend=bool(trend) and cfg.ETS_LEVEL_DAMPED,
            seasonal="add" if seasonal_periods else None, seasonal_periods=seasonal_periods,
        ).fit(disp=False)
        forecast = np.asarray(model.forecast(h)) * scale
    return forecast, len(caught)


def _flows_model(inflow: pd.Series, outflow: pd.Series, last_balance: float, h: int) -> tuple[np.ndarray, int, dict]:
    """Saldo proyectado a partir de ETS de entradas y salidas; la estacionalidad se identifica con el entrenamiento."""
    path, warns, periods = np.zeros(h), 0, {}
    for name, s, sign in (("inflow", inflow, 1.0), ("outflow", outflow, -1.0)):
        period = seasonality.identify_period(s)
        f, w = _ets(s.to_numpy(float), h, seasonal_periods=period)
        path += sign * np.clip(f, 0, None)
        warns += w
        periods[name] = period
    return last_balance + np.cumsum(path), warns, periods


def cv_level_forecasts(
    panel: pd.DataFrame, raw_naive: pd.DataFrame, n_splits: int = cfg.TSCV_N_SPLITS,
    test_size: int = cfg.TSCV_TEST_SIZE, gap: int = cfg.TSCV_GAP,
) -> tuple[pd.DataFrame, dict]:
    """Corre los modelos en cada corte de ``TimeSeriesSplit``.

    ``raw_naive`` = filas que ve el squad (``squad_pipeline(...)["balances"]``). Devuelve ``(detalle, info)``:
    ``detalle`` en formato largo (corte, origen, cuenta, modelo, h, pronóstico, real, error) e ``info`` con
    ``n_warnings`` por modelo y los periodos que identificó ``ets_flujos`` en cada corte.
    """
    bal = panel.pivot(index="date", columns="account_id", values="balance").sort_index()
    inflow = panel.pivot(index="date", columns="account_id", values="inflow").sort_index()
    outflow = panel.pivot(index="date", columns="account_id", values="outflow").sort_index()
    rows_by_acc = {a: g.sort_values("date", kind="stable") for a, g in raw_naive.groupby("account_id")}
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    out, warns, periods = [], {m: 0 for m in MODELS}, []
    for fold, (train, test) in enumerate(splitter.split(bal.index), start=1):
        origin = bal.index[train[-1]]
        steps = (test - train[-1]).astype(int)                       # h desde el origen (con gap, no arranca en 1)
        H = int(steps.max())
        for acc in bal.columns:
            y = bal[acc].iloc[train]
            actual = bal[acc].to_numpy()[test]
            preds = {
                "squad_sin_limpiar": np.full(len(test), squad_raw_mean(rows_by_acc[acc], origin)),
                "squad_limpio": np.full(len(test), y.iloc[-cfg.SQUAD_WINDOW:].mean()),
                "ultimo_saldo": np.full(len(test), y.iloc[-1]),
            }
            f, w = _ets(y.to_numpy(float), H, trend="add")
            preds["ets_nivel"], warns["ets_nivel"] = f[steps - 1], warns["ets_nivel"] + w
            f, w = _ets(y.to_numpy(float), H, trend="add", seasonal_periods=seasonality.WEEK)
            preds["ets_nivel_semanal"], warns["ets_nivel_semanal"] = f[steps - 1], warns["ets_nivel_semanal"] + w
            f, w, per = _flows_model(inflow[acc].iloc[train], outflow[acc].iloc[train], float(y.iloc[-1]), H)
            preds["ets_flujos"], warns["ets_flujos"] = f[steps - 1], warns["ets_flujos"] + w
            periods.append({"fold": fold, "account_id": acc, **{f"periodo_{k}": v for k, v in per.items()}})
            for model, pred in preds.items():
                out.append(pd.DataFrame({
                    "fold": fold, "origen": origin, "account_id": acc, "model": model, "h": steps,
                    "date": bal.index[test], "forecast": pred, "real": actual, "error": pred - actual,
                }))
    return pd.concat(out, ignore_index=True), {"n_warnings": warns, "periodos_ets_flujos": pd.DataFrame(periods)}


def cv_summary(detail: pd.DataFrame, horizons=cfg.HORIZONS) -> dict[str, pd.DataFrame]:
    """Resúmenes del detalle. MAE en unidades de la cuenta; para comparar entre cuentas se usa la razón contra ``BASE``.

    * ``por_cuenta``: MAE del bloque, MAE a cada ``h`` de ``horizons``, sesgo (+ = sobreestima) y razón vs ``BASE``.
    * ``por_corte``: razón media (entre cuentas) vs ``BASE`` en cada corte: mide si el resultado es estable en el tiempo.
    * ``global``: razón media entre cuentas y en cuántas cuentas el modelo le gana a ``BASE``.
    """
    d = detail.assign(abs_error=detail["error"].abs())
    per = d.groupby(["account_id", "model"]).agg(mae_bloque=("abs_error", "mean"), sesgo=("error", "mean"))
    for h in horizons:
        per[f"mae_h{h}"] = d[d["h"] == h].groupby(["account_id", "model"])["abs_error"].mean()
    base = per["mae_bloque"].xs(BASE, level="model")
    per["razon_vs_base"] = per["mae_bloque"] / per.index.get_level_values("account_id").map(base).to_numpy()
    per = per.reindex(MODELS, level="model").sort_index(level="account_id", sort_remaining=False)

    fold = d.groupby(["fold", "account_id", "model"])["abs_error"].mean()
    fold_ratio = fold / fold.xs(BASE, level="model").reindex(fold.index.droplevel("model")).to_numpy()
    by_fold = fold_ratio.groupby(["fold", "model"]).mean().unstack("model")[list(MODELS)]
    by_fold.insert(0, "origen", d.groupby("fold")["origen"].first())

    ratios = per["razon_vs_base"].unstack("model")[list(MODELS)]
    glob = pd.DataFrame({"razon_media_vs_base": ratios.mean(), "cuentas_que_mejoran_a_base": (ratios < 1).sum()})
    return {"por_cuenta": per, "por_corte": by_fold, "global": glob}
