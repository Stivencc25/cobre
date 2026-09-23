"""
seasonality.py
Diagnóstico de estacionariedad (ADF + KPSS) y estacionalidad para un panel de series
(p. ej. balance / inflow / outflow por account_id).

Uso:
    import seasonality
    diff_t, seas_t, stationary = seasonality.acf_pacf_report(panel)
    seasonality.resumen_panel(diff_t)
    seasonality.plot_acf_pacf(panel, "ACC-003", "inflow")

Formatos de `panel` aceptados:
    - Ancho: columnas [account_id, fecha, balance, inflow, outflow, ...]
    - Largo: columnas [account_id, fecha, serie, valor]
    - account_id / fecha pueden venir en el índice (se hace reset_index).

Reglas de decisión (alpha = 0.05):
    ADF  H0: raíz unitaria   -> p < alpha  => estacionaria
    KPSS H0: estacionaria    -> p < alpha  => NO estacionaria
    phi_impl = 1 + 2*ACF(1) de la diferencia  (persistencia AR(1) implícita del nivel)
      - ACF(1) de la diferencia ~ -0.5  -> nivel ~ ruido blanco -> no diferenciar
      - ACF(1) de la diferencia ~  0    -> nivel ~ caminata aleatoria -> diferenciar
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss

ALPHA = 0.05
UMBRAL_ACF1_DIF = -0.35   # por debajo: diferenciar sobra (nivel ~ ruido blanco)
PHI_RAIZ = 0.80           # persistencia implícita por encima de la cual se trata como I(1)
PHI_ESTAC = 0.60          # persistencia implícita por debajo de la cual se trata como I(0)


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------
def _to_long(panel: pd.DataFrame, id_col: str, t_col: str, value_cols=None) -> pd.DataFrame:
    df = panel.copy()
    if id_col not in df.columns or t_col not in df.columns:
        df = df.reset_index()
    if {"serie", "valor"}.issubset(df.columns):
        long = df[[id_col, t_col, "serie", "valor"]].copy()
    else:
        if value_cols is None:
            value_cols = [c for c in df.select_dtypes("number").columns if c not in (id_col, t_col)]
        long = df.melt(id_vars=[id_col, t_col], value_vars=value_cols,
                       var_name="serie", value_name="valor")
    long[t_col] = pd.to_datetime(long[t_col])
    return long.sort_values([id_col, "serie", t_col]).reset_index(drop=True)


def transformar(x: np.ndarray, metodo: str | None) -> np.ndarray:
    """None | 'log1p' | 'asinh'. log1p cae a asinh si hay negativos (p. ej. balance)."""
    x = np.asarray(x, dtype=float)
    if metodo is None or metodo == "ninguna":
        return x
    if metodo == "log1p":
        return np.arcsinh(x) if np.nanmin(x) < 0 else np.log1p(x)
    if metodo == "asinh":
        return np.arcsinh(x)
    raise ValueError(f"Transformación no soportada: {metodo}")


def _metodo(transform, serie):
    return transform.get(serie) if isinstance(transform, dict) else transform


def _adf(x, regression):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return float(adfuller(x, regression=regression, autolag="AIC")[1])
    except Exception:
        return np.nan


def _kpss(x, regression):
    # p-valor interpolado en tablas: truncado en [0.01, 0.10]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            return float(kpss(x, regression=regression, nlags="auto")[1])
    except Exception:
        return np.nan


def _acf_k(x, k):
    try:
        return float(acf(x, nlags=k, fft=False, missing="drop")[k])
    except Exception:
        return np.nan


# --------------------------------------------------------------------------------------
# Decisión de d
# --------------------------------------------------------------------------------------
def _decidir(adf_c, kpss_c, adf_ct, kpss_ct, phi, alpha=ALPHA):
    raiz_c = (adf_c >= alpha) and (kpss_c < alpha)
    raiz_ct = (adf_ct >= alpha) and (kpss_ct < alpha)
    est_c = (adf_c < alpha) and (kpss_c >= alpha)
    est_ct = (adf_ct < alpha) and (kpss_ct >= alpha)

    # 1) Ambas pruebas + persistencia alta coinciden en raíz unitaria
    if phi > PHI_RAIZ and (raiz_c or raiz_ct or adf_ct >= alpha):
        return 1, "raíz unitaria"
    # 2) Ambas pruebas coinciden en estacionariedad
    if est_c:
        return 0, "estacionaria"
    # 3) Estacionaria alrededor de tendencia determinística
    if est_ct:
        return 0, "estacionaria en tendencia"
    # 4) Pruebas en conflicto: desempata la persistencia implícita
    if phi < PHI_ESTAC:
        return 0, "estacionaria c/ quiebre o estacionalidad"
    if phi > PHI_RAIZ:
        return 1, "desempate ACF(1): raíz unitaria"
    return (1 if raiz_c else 0), "inconcluso"


def diagnosticar_serie(x, alpha=ALPHA, lags_s=(7,), min_obs=24):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    base = {"n": n}
    if n < min_obs or np.nanstd(x) == 0:
        base.update({"d": np.nan, "caso": "insuficiente / constante"})
        return base, {}

    adf_c, adf_ct = _adf(x, "c"), _adf(x, "ct")
    kpss_c, kpss_ct = _kpss(x, "c"), _kpss(x, "ct")

    dx = np.diff(x)
    acf1_niv = _acf_k(x, 1)
    acf1_dif = _acf_k(dx, 1)
    phi = float(np.clip(1 + 2 * acf1_dif, -1, 1.2))

    d, caso = _decidir(adf_c, kpss_c, adf_ct, kpss_ct, phi, alpha)

    adf_raiz = adf_c >= alpha
    kpss_raiz = kpss_c < alpha

    diag = {
        **base,
        "adf_p": adf_c, "kpss_p": kpss_c,
        "adf_p_ct": adf_ct, "kpss_p_ct": kpss_ct,
        "kpss_truncado": (kpss_c <= 0.01) or (kpss_c >= 0.10),
        "acf1_nivel": acf1_niv,
        "acf1_dif": acf1_dif,
        "phi_impl": phi,
        "d": d,
        "caso": caso,
        "kpss_confirma": adf_raiz == kpss_raiz,
        "sobrediferenciada": (d == 1) and (acf1_dif < UMBRAL_ACF1_DIF),
        "dif_innecesaria": (d == 0) and (acf1_dif < UMBRAL_ACF1_DIF),
    }

    # Estacionalidad sobre la serie ya estacionaria (diferenciada si d = 1)
    z = dx if d == 1 else x
    banda = 1.96 / np.sqrt(len(z))
    seas = {"n_est": len(z), "banda_95": banda}
    for s in lags_s:
        if len(z) > 2 * s:
            acf_s = _acf_k(z, s)
            try:
                lb_p = float(acorr_ljungbox(z, lags=[s], return_df=True)["lb_pvalue"].iloc[0])
            except Exception:
                lb_p = np.nan
            seas[f"acf_{s}"] = acf_s
            seas[f"lb_p_{s}"] = lb_p
            seas[f"estacional_{s}"] = abs(acf_s) > banda
        else:
            seas[f"acf_{s}"] = np.nan
            seas[f"lb_p_{s}"] = np.nan
            seas[f"estacional_{s}"] = False
    return diag, seas


# --------------------------------------------------------------------------------------
# Reporte de panel
# --------------------------------------------------------------------------------------
def acf_pacf_report(panel: pd.DataFrame,
                    id_col: str = "account_id",
                    t_col: str = "fecha",
                    value_cols=None,
                    transform=None,
                    alpha: float = ALPHA,
                    lags_s=(7,),
                    min_obs: int = 24,
                    tipo_serie: dict | None = None,
                    umbral_voto: float = 0.5):
    """
    Parameters
    ----------
    transform  : None | 'log1p' | 'asinh' | dict por serie, p. ej.
                 {"balance": "asinh", "inflow": "log1p", "outflow": "log1p"}
    lags_s     : rezagos estacionales a revisar (7 diario-semanal, 2 quincenal, 12 mensual-anual)
    tipo_serie : opcional {"balance": "stock", "inflow": "flujo", ...}; fija d_panel
                 (stock -> 1, flujo -> 0) y reporta si las pruebas coinciden.
    umbral_voto: fracción de cuentas con d=1 para diferenciar la serie en todo el panel.

    Returns
    -------
    diff_t     : diagnóstico por (cuenta, serie)
    seas_t     : estacionalidad por (cuenta, serie)
    stationary : panel ancho [id, fecha, series...] transformado con d_panel aplicado
    """
    long = _to_long(panel, id_col, t_col, value_cols)

    rows, srows = [], []
    for (acc, serie), g in long.groupby([id_col, "serie"], sort=True):
        met = _metodo(transform, serie)
        xt = transformar(g["valor"].to_numpy(), met)
        diag, seas = diagnosticar_serie(xt, alpha=alpha, lags_s=lags_s, min_obs=min_obs)
        rows.append({id_col: acc, "serie": serie, "transform": met or "ninguna", **diag})
        srows.append({id_col: acc, "serie": serie, **seas})

    diff_t = pd.DataFrame(rows).set_index([id_col, "serie"])
    seas_t = pd.DataFrame(srows).set_index([id_col, "serie"])

    # Decisión común por serie (voto entre cuentas), más robusta que cuenta por cuenta
    frac_d1 = diff_t.groupby(level="serie")["d"].mean()
    d_voto = (frac_d1 >= umbral_voto).astype(int)
    series_idx = diff_t.index.get_level_values("serie")
    diff_t["frac_d1_serie"] = series_idx.map(frac_d1)

    if tipo_serie:
        d_tipo = {s: (1 if t == "stock" else 0) for s, t in tipo_serie.items()}
        d_panel = pd.Series({s: d_tipo.get(s, d_voto.get(s)) for s in d_voto.index})
        diff_t["coincide_tipo"] = diff_t["d"] == series_idx.map(d_panel)
    else:
        d_panel = d_voto
    diff_t["d_panel"] = series_idx.map(d_panel)

    # Panel estacionario listo para modelar
    partes = []
    for (acc, serie), g in long.groupby([id_col, "serie"], sort=True):
        xt = transformar(g["valor"].to_numpy(), _metodo(transform, serie))
        s = pd.Series(xt, index=g[t_col].values)
        if d_panel.get(serie, 0) == 1:
            s = s.diff()
        partes.append(pd.DataFrame({id_col: acc, t_col: s.index, "serie": serie, "valor": s.values}))
    stationary = (pd.concat(partes)
                  .pivot_table(index=[id_col, t_col], columns="serie", values="valor")
                  .reset_index())
    stationary.columns.name = None
    stationary = stationary.rename(columns={s: f"d_{s}" for s in d_panel.index if d_panel[s] == 1})

    return diff_t, seas_t, stationary


def resumen_panel(diff_t: pd.DataFrame) -> pd.DataFrame:
    """Resumen por serie: casos, % d=1, conflictos ADF/KPSS y decisión final."""
    g = diff_t.groupby(level="serie")
    res = pd.DataFrame({
        "cuentas": g.size(),
        "% d=1": g["d"].mean().round(2),
        "% ADF/KPSS coinciden": g["kpss_confirma"].mean().round(2),
        "ACF(1) dif mediana": g["acf1_dif"].median().round(2),
        "phi_impl mediana": g["phi_impl"].median().round(2),
        "% sobrediferenciada": g["sobrediferenciada"].mean().round(2),
        "% dif innecesaria": g["dif_innecesaria"].mean().round(2),
        "d_panel": g["d_panel"].first(),
    })
    casos = pd.crosstab(diff_t.index.get_level_values("serie"), diff_t["caso"])
    casos.index.name = "serie"
    return res.join(casos)


# --------------------------------------------------------------------------------------
# Gráfico
# --------------------------------------------------------------------------------------
def plot_acf_pacf(panel, cuenta, serie, id_col="account_id", t_col="fecha",
                  transform=None, lags=30):
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    long = _to_long(panel, id_col, t_col)
    g = long[(long[id_col] == cuenta) & (long["serie"] == serie)]
    x = pd.Series(transformar(g["valor"].to_numpy(), _metodo(transform, serie)),
                  index=g[t_col].values).dropna()
    dx = x.diff().dropna()
    lags = min(lags, len(dx) // 2 - 1)

    fig, ax = plt.subplots(3, 2, figsize=(13, 9))
    x.plot(ax=ax[0, 0], title=f"{cuenta} · {serie} (nivel)")
    dx.plot(ax=ax[0, 1], title="Primera diferencia")
    plot_acf(x, lags=lags, ax=ax[1, 0], title="ACF nivel")
    plot_pacf(x, lags=lags, ax=ax[2, 0], method="ywm", title="PACF nivel")
    plot_acf(dx, lags=lags, ax=ax[1, 1], title="ACF diferencia")
    plot_pacf(dx, lags=lags, ax=ax[2, 1], method="ywm", title="PACF diferencia")
    fig.tight_layout()
    return fig