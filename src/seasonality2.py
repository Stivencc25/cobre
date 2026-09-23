"""D4c extendido — versión más rica de ``seasonality.acf_pacf_report``.

Añade, sobre la decisión de diferenciar de ``seasonality.py``:
* **transformación por serie** (``log1p``, ``asinh``) antes de probar raíz unitaria, para estabilizar varianza
  en series con escalas muy distintas entre cuentas/monedas;
* **ADF y KPSS en dos especificaciones**, constante ('c') y constante+tendencia ('ct') — la 'ct' separa una serie
  con tendencia determinística (no necesita diferenciarse, basta con restarle la tendencia) de una con raíz
  unitaria de verdad (si necesita diferenciarse);
* **AR(1) implícito** (``phi_impl``): un número interpretable de qué tan persistente es la serie (cerca de 1 =
  casi no revierte a la media; bajo = revierte rápido). Es una regresión OLS simple separada, no el coeficiente
  interno del ADF aumentado (que puede llevar rezagos adicionales) — se reporta como resumen, no como el test.
* **clasificación de caso** cruzando ADF y KPSS (el contraste clásico: cuándo coinciden y cuándo no);
* **decisión agregada a nivel panel** (``d_panel``) combinando el p-valor de ADF de las 6 cuentas para la misma
  serie con el método de Fisher (test de panel tipo Maddala-Wu) — con series de ~270 días, un ADF por cuenta
  tiene poca potencia; combinar las 6 cuentas de la misma serie da un veredicto más confiable para la serie en
  conjunto. Ojo: rechazar la combinada solo dice "no todas tienen raíz unitaria" (alternativa heterogénea), no
  que ninguna la tenga — se reporta junto al detalle por cuenta, no en su lugar.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import config as cfg
from .seasonality import spike, stationary_version

TRANSFORMS = {
    "none": lambda x: x,
    "log1p": np.log1p,   # series no negativas (inflow/outflow): comprime la escala, define en 0
    "asinh": np.arcsinh,  # series de cualquier signo (balance): como log pero definido en 0 y negativos
}


def _apply_transform(s: pd.Series, name: str) -> pd.Series:
    return pd.Series(TRANSFORMS[name](s.to_numpy(float)), index=s.index)


def _adf_p(s: pd.Series, regression: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(s, autolag="AIC", regression=regression, result_object=False)[1])


def _kpss_p(s: pd.Series, regression: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kpss(s, regression=regression, nlags="auto")[1])


def implied_phi(s: pd.Series) -> float:
    """AR(1) implícito: OLS de ``y_t`` sobre ``y_{t-1}`` (con intercepto), phi = pendiente.

    Es una regresión de un solo rezago, separada de la regresión aumentada que usa ADF por dentro (que puede
    llevar más rezagos según AIC) — se reporta como un resumen interpretable, no como parte del test formal.
    """
    y = s.to_numpy(float)
    y_t, y_l = y[1:], y[:-1]
    X = np.column_stack([np.ones_like(y_l), y_l])
    beta, *_ = np.linalg.lstsq(X, y_t, rcond=None)
    return float(beta[1])


def classify_case(adf_c_p: float, kpss_c_p: float, adf_ct_p: float, alpha: float) -> str:
    """Cruza ADF-c y KPSS-c (el contraste clásico) y usa ADF-ct para separar tendencia determinística de raíz unitaria.

    - ADF-c rechaza, KPSS-c no rechaza -> coinciden: **estacionaria (nivel)**.
    - ADF-c no rechaza, KPSS-c rechaza -> coinciden en no-estacionaria; si ADF-ct SÍ rechaza, la no-estacionariedad
      es solo una tendencia determinística (**tendencia determinística**, no hace falta diferenciar, basta
      restar la tendencia); si ADF-ct tampoco rechaza, es **raíz unitaria (diferenciar)**.
    - Cualquier otra combinación (ambos rechazan o ninguno rechaza): **ambiguo (contraste ADF/KPSS)**.
    """
    adf_c_rej, kpss_c_rej, adf_ct_rej = adf_c_p < alpha, kpss_c_p < alpha, adf_ct_p < alpha
    if adf_c_rej and not kpss_c_rej:
        return "estacionaria (nivel)"
    if not adf_c_rej and kpss_c_rej:
        return "tendencia determinística" if adf_ct_rej else "raíz unitaria (diferenciar)"
    return "ambiguo (contraste ADF/KPSS)"


def differencing_decision2(s: pd.Series, alpha: float = cfg.ADF_ALPHA) -> dict:
    """Como ``seasonality.differencing_decision``, con ADF/KPSS en 'c' y 'ct', AR(1) implícito y el caso."""
    adf_p, adf_p_ct = _adf_p(s, "c"), _adf_p(s, "ct")
    kpss_p, kpss_p_ct = _kpss_p(s, "c"), _kpss_p(s, "ct")
    acf1_dif = float(acf(s.diff().dropna(), nlags=1, fft=True)[1])
    over = acf1_dif < cfg.OVERDIFF_ACF1
    d = int(adf_p > alpha and not over)
    caso = classify_case(adf_p, kpss_p, adf_p_ct, alpha)
    return {
        "adf_p": adf_p, "kpss_p": kpss_p, "adf_p_ct": adf_p_ct, "kpss_p_ct": kpss_p_ct,
        "acf1_dif": acf1_dif, "phi_impl": implied_phi(s), "d": d, "caso": caso,
        "kpss_confirma": bool(kpss_p < alpha) if d else bool(kpss_p >= alpha),
        "sobrediferenciada": bool(over),
        "dif_innecesaria": bool(d == 0 and caso == "estacionaria (nivel)"),
    }


def _fisher_combine(pvalues) -> float:
    """Método de Fisher (test de panel tipo Maddala-Wu): combina k p-valores de ADF (misma H0: raíz unitaria,
    una cuenta por unidad) en un único p-valor de panel. Estadístico ``-2*sum(ln p_i) ~ chi2(2k)`` bajo H0.
    """
    ps = np.clip(np.asarray(pvalues, dtype=float), 1e-12, 1.0)
    stat = -2.0 * np.log(ps).sum()
    return float(1 - stats.chi2.cdf(stat, df=2 * len(ps)))


def acf_pacf_report(panel: pd.DataFrame, id_col: str = "account_id", t_col: str = "date",
                     transform: dict[str, str] | None = None, lags_s: tuple[int, ...] = (7,),
                     tipo_serie: dict[str, str] | None = None, series: tuple[str, ...] = ("balance", "inflow", "outflow"),
                     alpha: float = cfg.ADF_ALPHA):
    """``diff_t, seas_t, stationary`` — igual forma que ``seasonality.acf_pacf_report``, con las columnas nuevas
    de ``differencing_decision2`` más ``tipo_serie`` (informativo) y ``d_panel`` (decisión agregada de panel).
    """
    transform, tipo_serie = transform or {}, tipo_serie or {}
    rows_d, rows_s, stationary = [], [], {}
    adf_p_by_series: dict[str, list[float]] = {name: [] for name in series}

    for acc, g in panel.sort_values(t_col).groupby(id_col):
        g = g.set_index(t_col).asfreq("D")
        for col in series:
            s = _apply_transform(g[col].astype(float), transform.get(col, "none"))
            dec = differencing_decision2(s, alpha=alpha)
            adf_p_by_series[col].append(dec["adf_p"])

            x = stationary_version(s, dec["d"])
            stationary[(acc, col)] = x
            band = 1.96 / np.sqrt(len(x))
            a, p = acf(x, nlags=cfg.ACF_NLAGS, fft=True), pacf(x, nlags=cfg.ACF_NLAGS, method="ywm")
            extra = {f"pico_{lag}": spike(a, lag) for lag in lags_s if lag != 7 and lag <= cfg.ACF_NLAGS}
            pico_7 = spike(a, 7)

            rows_d.append({"account_id": acc, "serie": col, "tipo_serie": tipo_serie.get(col, ""), **dec})
            rows_s.append({
                "account_id": acc, "serie": col, "d": dec["d"], "acf_7": a[7], "pacf_7": p[7], "pico_7": pico_7,
                "picos_estacionales": int(sum(v > band for v in (pico_7, *extra.values()))), **extra,
            })

    diff_t = pd.DataFrame(rows_d).set_index(["account_id", "serie"])
    seas_t = pd.DataFrame(rows_s).set_index(["account_id", "serie"])
    d_panel = {col: int(_fisher_combine(ps) > alpha) for col, ps in adf_p_by_series.items()}
    diff_t["d_panel"] = diff_t.index.get_level_values("serie").map(d_panel)
    return diff_t, seas_t, stationary


def resumen_panel(diff_t: pd.DataFrame) -> pd.DataFrame:
    """Una fila por serie: cuántas cuentas piden diferenciar individualmente vs. la decisión agregada de panel."""
    g = diff_t.groupby("serie")
    return pd.DataFrame({
        "cuentas que difieren (d=1)": g["d"].sum().astype(int),
        "de": g["d"].size(),
        "d_panel (Fisher)": g["d_panel"].first(),
        "caso más común": g["caso"].agg(lambda s: s.value_counts().idxmax()),
    })


def plot_acf_pacf(panel: pd.DataFrame, account_id: str, serie: str, transform: str = "none",
                   id_col: str = "account_id", t_col: str = "date"):
    """ACF/PACF de una sola cuenta y serie, con su transformación — para revisar un caso puntual sin recalcular
    la grilla completa de ``acf_pacf_report``.
    """
    import matplotlib.pyplot as plt

    g = panel[panel[id_col] == account_id].sort_values(t_col).set_index(t_col).asfreq("D")
    s = _apply_transform(g[serie].astype(float), transform)
    dec = differencing_decision2(s)
    x = stationary_version(s, dec["d"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    band = 1.96 / np.sqrt(len(x))
    for ax, (vals, name) in zip(axes, ((acf(x, nlags=cfg.ACF_NLAGS, fft=True)[1:], "ACF"),
                                        (pacf(x, nlags=cfg.ACF_NLAGS, method="ywm")[1:], "PACF"))):
        lags = np.arange(1, len(vals) + 1)
        ax.axhspan(-band, band, color="0.9", zorder=1)
        ax.bar(lags, vals, color="#2a78d6", zorder=2, width=0.65)
        ax.axhline(0, color="k", lw=0.8, zorder=2)
        ax.set_title(f"{account_id} · {serie} ({transform}) · {name}", fontsize=9)
    return fig
