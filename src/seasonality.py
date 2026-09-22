"""D4b — Identificación de la estacionalidad semanal: ¿existe, en qué serie y con qué fuerza?

El saldo es el flujo neto acumulado, así que su estacionalidad equivale a la del ``net_flow``: por eso se analizan
``inflow``, ``outflow`` y ``net_flow``. La decisión ("hay patrón semanal") la toma un test de hipótesis
(Kruskal-Wallis entre los 7 días de la semana, nivel ``cfg.SEASONAL_ALPHA``); la fuerza STL, la autocorrelación en el
rezago 7 y el periodograma son evidencia de apoyo, no criterio de decisión.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from . import config as cfg

WEEK = 7
SERIES = ("inflow", "outflow", "net_flow")


def weekday_test(y: pd.Series) -> float:
    """p-valor de Kruskal-Wallis entre los 7 días de la semana (``y`` indexada por fecha)."""
    groups = [y[y.index.dayofweek == d].dropna() for d in range(WEEK)]
    return float(stats.kruskal(*groups).pvalue)


def identify_period(y: pd.Series, alpha: float = cfg.SEASONAL_ALPHA) -> int | None:
    """7 si la serie tiene efecto de día de semana significativo, ``None`` si no.

    Se llama con datos de entrenamiento únicamente, para que la decisión no vea el bloque de prueba.
    """
    return WEEK if weekday_test(y) < alpha else None


def seasonal_strength(y: pd.Series, period: int = WEEK) -> float:
    """Fuerza estacional de Hyndman: ``max(0, 1 - Var(resid) / Var(estacional + resid))`` sobre un STL robusto.

    Ojo: STL siempre extrae algo de "estacionalidad", incluso del ruido. Con n = 270 y periodo 7, el ruido blanco da
    ~0,20 (p95 ~0,29): un valor en ese rango no es evidencia. Por eso no se usa para decidir.
    """
    r = STL(y.to_numpy(float), period=period, robust=True).fit()
    return float(max(0.0, 1 - np.var(r.resid) / np.var(r.seasonal + r.resid)))


def dominant_period(y: pd.Series) -> float:
    """Periodo (en días) del pico más alto del periodograma, tras quitar la tendencia lineal."""
    f, p = signal.periodogram(signal.detrend(y.to_numpy(float)))
    return float(1 / f[1:][np.argmax(p[1:])])


def weekly_seasonality_report(panel: pd.DataFrame, alpha: float = cfg.SEASONAL_ALPHA) -> pd.DataFrame:
    """Evidencia por cuenta y serie. ``semanal`` = el test de día de semana rechaza "sin efecto" a nivel ``alpha``."""
    rows = []
    for acc, g in panel.sort_values("date").groupby("account_id"):
        g = g.set_index("date")
        for name in SERIES:
            y = g[name].astype(float)
            band = 1.96 / np.sqrt(len(y))
            p = weekday_test(y)
            rows.append({
                "account_id": acc, "serie": name, "fuerza_stl_7": seasonal_strength(y),
                "acf_rezago_7": float(acf(y - y.mean(), nlags=WEEK, fft=True)[WEEK]), "banda_95": band,
                "periodo_dominante": dominant_period(y), "kruskal_p": p, "semanal": p < alpha,
            })
    return pd.DataFrame(rows).set_index(["account_id", "serie"])


# ------------------------------------------------------------------ ACF / PACF y decisión de diferenciar
import warnings  # noqa: E402

from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.seasonal import MSTL  # noqa: E402
from statsmodels.tsa.stattools import adfuller, kpss, pacf  # noqa: E402

ACF_SERIES = ("balance", "inflow", "outflow")


def differencing_decision(s: pd.Series) -> dict:
    """¿Diferenciar? Sí solo si ADF no rechaza raíz unitaria Y la diferencia no queda sobrediferenciada.

    Sobrediferenciada = ACF(1) de la diferencia por debajo de ``cfg.OVERDIFF_ACF1`` (firma de diferenciar lo que no lo
    pedía: cerca de -0,5; el punto medio entre 0 y -0,5 es más robusto que la banda de ruido blanco, que con n = 270
    (±0,12) confundía una autocorrelación leve, -0,13, con sobrediferenciación). KPSS se reporta como contraste; sus
    p-valores están truncados a [0,01; 0,10] por el propio test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(s, autolag="AIC", result_object=False)[1]
        kpss_p = kpss(s, regression="c", nlags="auto", result_object=False)[1]
    acf1 = float(acf(s.diff().dropna(), nlags=1, fft=True)[1])
    over = acf1 < cfg.OVERDIFF_ACF1
    d = int(adf_p > cfg.ADF_ALPHA and not over)
    return {"adf_p": float(adf_p), "kpss_p": float(kpss_p), "acf1_dif": acf1, "sobrediferenciada": bool(over), "d": d,
            "kpss_confirma": bool(kpss_p < 0.05) if d else bool(kpss_p >= 0.05)}


def stationary_version(s: pd.Series, d: int) -> pd.Series:
    """Serie lista para leer el ACF: primera diferencia si ``d = 1``; si no, el nivel menos la tendencia MSTL."""
    if d:
        return s.diff().dropna()
    trend = MSTL(s, periods=list(cfg.MSTL_PERIODS), stl_kwargs={"robust": True}).fit().trend
    return s - trend


def spike(a: np.ndarray, lag: int, nlags: int = cfg.ACF_NLAGS) -> float:
    """ACF en un rezago estacional menos el promedio de sus vecinos no estacionales (lag±1, lag±2): cuánto sobresale."""
    return float(a[lag] - np.mean([a[lag + k] for k in (-2, -1, 1, 2) if lag + k <= nlags]))


def acf_pacf_report(panel: pd.DataFrame):
    """ACF/PACF por cuenta y serie (saldo, inflow, outflow). Devuelve ``(decisión_d, estacionalidad, estacionarias)``.

    ``estacionarias`` = dict ``(cuenta, serie) -> serie estacionaria`` para graficar el ACF/PACF. Se asume el panel ya
    limpio (F1): sobre el crudo, las lecturas corruptas distorsionan el ACF y hay que enmascararlas antes.
    """
    rows_d, rows_s, stationary = [], [], {}
    for acc, g in panel.sort_values("date").groupby("account_id"):
        g = g.set_index("date").asfreq("D")
        for col in ACF_SERIES:
            s = g[col].astype(float)
            dec = differencing_decision(s)
            x = stationary_version(s, dec["d"])
            stationary[(acc, col)] = x
            band = 1.96 / np.sqrt(len(x))
            a, p = acf(x, nlags=cfg.ACF_NLAGS, fft=True), pacf(x, nlags=cfg.ACF_NLAGS, method="ywm")
            spikes = [spike(a, lag) for lag in cfg.SEASONAL_LAGS]
            rows_d.append({"account_id": acc, "serie": col, **dec})
            rows_s.append({
                "account_id": acc, "serie": col, "d": dec["d"], "acf_7": a[7], "pacf_7": p[7], "acf_14": a[14], "acf_21": a[21],
                "pico_7": spikes[0], "picos_estacionales": int(sum(sp > band for sp in spikes)),
                "ljung_box_p7": float(acorr_ljungbox(x, lags=[7])["lb_pvalue"].iloc[0]),
                "acf_7_tras_dif_7": float(acf(x.diff(7).dropna(), nlags=7, fft=True)[7]),
            })
    diff_t = pd.DataFrame(rows_d).set_index(["account_id", "serie"])
    seas_t = pd.DataFrame(rows_s).set_index(["account_id", "serie"])
    band270 = 1.96 / np.sqrt(len(panel[panel["account_id"] == panel["account_id"].iloc[0]]))
    seas_t["semanal"] = (seas_t["pico_7"] > band270) & (seas_t["picos_estacionales"] >= 2)
    seas_t["D"] = ((seas_t["acf_7"] > 0.5) & (seas_t["acf_7_tras_dif_7"] > -0.25)).astype(int)
    return diff_t, seas_t, stationary


# ------------------------------------------------ perfiles por ciclo (para VER la estacionalidad antes de probarla)
CYCLES = ("semanal", "quincenal", "mensual")
DETREND_WINDOW = 28          # media centrada de 28 días (múltiplo de 7: no filtra el patrón semanal)


def detrended_pct(s: pd.Series, window: int = DETREND_WINDOW) -> pd.Series:
    """Desviación % de la serie respecto de su media móvil centrada: quita la tendencia para ver solo el patrón del ciclo."""
    trend = s.rolling(window, center=True, min_periods=window // 2).mean()
    return (s / trend - 1) * 100


def cycle_position(index: pd.DatetimeIndex, cycle: str) -> np.ndarray:
    """Posición de cada fecha dentro del ciclo: día de la semana (0=lun), día de la quincena (1-16) o día del mes (1-31)."""
    if cycle == "semanal":
        return index.dayofweek.to_numpy()
    if cycle == "quincenal":
        d = index.day.to_numpy()
        return np.where(d <= 15, d, d - 15)
    if cycle == "mensual":
        return index.day.to_numpy()
    raise ValueError(f"ciclo desconocido: {cycle!r}")


def cycle_profile(s: pd.Series, cycle: str, window: int = DETREND_WINDOW) -> tuple[pd.DataFrame, float]:
    """Perfil medio de la serie sin tendencia por posición del ciclo, con error estándar, y p-valor de Kruskal-Wallis.

    Devuelve ``(tabla[pos, media, se, n], p)``. Con ~9 meses de datos: cada día de la semana tiene ~38 observaciones, cada
    posición de la quincena ~17 y cada día del mes ~9, así que el ciclo mensual es el menos fiable. Un ciclo "mes del año"
    no es estimable con menos de un año de historia.
    """
    x = detrended_pct(s.astype(float), window).dropna()
    pos = cycle_position(x.index, cycle)
    g = x.groupby(pos)
    tab = pd.DataFrame({"media": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.size()), "n": g.size()}).rename_axis("pos").reset_index()
    groups = [v.to_numpy() for _, v in g if len(v) >= 2]
    p = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan
    return tab, p


def cycle_pvalues(panel: pd.DataFrame, series=ACF_SERIES if "ACF_SERIES" in globals() else ("balance", "inflow", "outflow")) -> pd.DataFrame:
    """p-valor de Kruskal-Wallis por cuenta, serie y ciclo (semanal / quincenal / mensual). Menor = más evidencia de patrón."""
    rows = []
    for acc, g in panel.sort_values("date").groupby("account_id"):
        g = g.set_index("date")
        for col in series:
            rows.append({"account_id": acc, "serie": col, **{c: cycle_profile(g[col], c)[1] for c in CYCLES}})
    return pd.DataFrame(rows).set_index(["account_id", "serie"])
