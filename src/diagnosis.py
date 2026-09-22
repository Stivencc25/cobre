"""Parte 1 — Evidencia del diagnóstico: reproduce el pipeline del squad y mide cada defecto (D1–D10)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .forecast import OriginForecast


# ----------------------------------------------------- pipeline del squad, tal cual
def squad_pipeline(accounts: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Reproduce EXACTAMENTE el notebook del squad sobre el crudo (coerce + dropna, media 14 filas, idxmax)."""
    balances = raw.copy()
    balances["date"] = pd.to_datetime(balances["date"], errors="coerce")
    balances = balances.dropna(subset=["date"]).sort_values(["account_id", "date"])
    forecasts = {acc: balances[balances.account_id == acc].sort_values("date")["balance"].tail(cfg.SQUAD_WINDOW).mean()
                 for acc in accounts.account_id}
    latest = balances.sort_values("date").groupby("account_id")["balance"].last()
    recs = []
    for acc, thr in cfg.SQUAD_THRESHOLD.items():
        if forecasts[acc] < thr:
            donor = latest.drop(index=acc).idxmax()
            recs.append({"to_account": acc, "from_account": donor, "amount_needed": round(thr - forecasts[acc], 2), "transfer_today": True})
    return {"balances": balances, "forecasts": forecasts, "latest": latest, "recommendations": recs}


def parse_loss_by_format(raw: pd.DataFrame) -> pd.DataFrame:
    from .quality import date_format_counts

    naive_ok = pd.to_datetime(raw["date"], errors="coerce").notna()
    fmt = raw["date"].map(lambda v: next((f for p, f in cfg.DATE_FORMATS if __import__("re").fullmatch(p, v)), "?"))
    out = pd.DataFrame({"formato": fmt, "sobrevive_al_squad": naive_ok}).groupby("formato")["sobrevive_al_squad"].agg(["size", "sum"])
    out.columns = ["filas", "sobreviven"]
    out["perdidas"] = out["filas"] - out["sobreviven"]
    return out


# --------------------------------------------------------- D2: contaminación del forecast
def squad_forecast_series(balance: pd.DataFrame, by: str) -> pd.DataFrame:
    """Media móvil de 14 observaciones (filas) por cuenta, indexada por la fecha de la última fila."""
    g = balance.sort_values(["account_id", by])
    g = g.assign(squad_fc=g.groupby("account_id")["balance"].transform(lambda s: s.rolling(cfg.SQUAD_WINDOW).mean()))
    return g[["account_id", by, "squad_fc"]].rename(columns={by: "date"}).dropna()


def contamination_table(raw_naive: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Cuánto se aleja el forecast del squad sobre datos crudos del mismo forecast sobre datos limpios."""
    naive = squad_forecast_series(raw_naive.rename(columns={"date": "date"}), "date").groupby(["account_id", "date"], as_index=False).last()
    clean = squad_forecast_series(panel, "date")
    m = naive.merge(clean, on=["account_id", "date"], suffixes=("_squad", "_clean"))
    m["desvio_pct"] = (m["squad_fc_squad"] / m["squad_fc_clean"] - 1) * 100
    rows = []
    for acc, g in m.groupby("account_id"):
        thr = cfg.SQUAD_THRESHOLD.get(acc)
        rows.append(
            {
                "account_id": acc, "dias_comparados": len(g),
                "dias_con_desvio>10%": int((g["desvio_pct"].abs() > 10).sum()),
                "desvio_max_%": float(g["desvio_pct"].abs().max()),
                "alertas_con_datos_squad": int((g["squad_fc_squad"] < thr).sum()) if thr else np.nan,
                "alertas_con_datos_limpios": int((g["squad_fc_clean"] < thr).sum()) if thr else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("account_id")


# ------------------------------------------------- D4/D5: forecast de nivel vs alternativas
def level_forecast_backtest(panel: pd.DataFrame, store, horizons=cfg.HORIZONS, start: int = cfg.MIN_TRAIN_DAYS - 1, n_draws: int = 200):
    """Saldo a h días: media 14d del nivel (squad) vs último saldo vs mediana de la distribución propuesta.

    Devuelve ``(tabla_por_cuenta, serie_para_grafico)``. La serie del squad es un escalar constante en el
    horizonte (sin drift, sin estacionalidad); aquí se mide cuánto rezaga y con qué sesgo.
    """
    bal = panel.pivot(index="date", columns="account_id", values="balance").sort_index()
    ccy = panel.drop_duplicates("account_id").set_index("account_id")["currency"]
    T = len(bal)
    rows, series = [], []
    for h in horizons:
        for a, acc in enumerate(bal.columns):
            b = bal[acc].to_numpy()
            err = {"squad": [], "ultimo": [], "propuesto": []}
            for t in range(start, T - h):
                squad = b[t - cfg.SQUAD_WINDOW + 1 : t + 1].mean()
                f: OriginForecast = store[t]
                med = b[t] + np.median(np.cumsum(f.draw(n_draws, np.random.default_rng([cfg.BOOTSTRAP_SEED, t]))[:, :h, a], axis=1)[:, -1])
                real = b[t + h]
                err["squad"].append(squad - real); err["ultimo"].append(b[t] - real); err["propuesto"].append(med - real)
                if h == 7:
                    series.append({"account_id": acc, "currency": ccy[acc], "date": bal.index[t + h], "realized": real, "squad": squad, "last": b[t]})
            rows.append(
                {"account_id": acc, "h": h, **{f"mae_{k}": np.abs(v).mean() for k, v in err.items()},
                 **{f"sesgo_{k}": np.mean(v) for k, v in err.items()}}
            )
    tab = pd.DataFrame(rows)
    for k in ("squad", "ultimo", "propuesto"):
        tab[f"mae_{k}_rel"] = tab[f"mae_{k}"] / tab["mae_squad"]
    return tab, pd.DataFrame(series)


def squad_raw_mean(rows: pd.DataFrame, day, window: int = cfg.SQUAD_WINDOW) -> float:
    """Forecast del squad al día ``day`` sobre sus filas SIN limpiar: media de las últimas ``window`` filas con fecha <= day.

    ``rows`` = filas de UNA cuenta ordenadas por fecha (``squad_pipeline(...)["balances"]``); ``mean`` omite nulos,
    igual que su ``forecast_next_week``. Sin lookahead: solo mira filas hasta ``day``.
    """
    k = int(np.searchsorted(rows["date"].to_numpy(), np.datetime64(day), side="right"))
    return float(np.nanmean(rows["balance"].to_numpy()[max(0, k - window):k]))


def old_vs_new_forecast(raw_naive: pd.DataFrame, panel: pd.DataFrame, store, horizons=cfg.HORIZONS,
                        start: int = cfg.MIN_TRAIN_DAYS - 1, n_draws: int = 200):
    """Forecast del squad sobre datos SIN limpiar vs. sobre datos limpios vs. método propuesto sobre datos limpios.

    Los tres se evalúan contra el saldo real del panel limpio, con el criterio de :func:`level_forecast_backtest`.
    ``raw_naive`` = filas que ve el squad (``squad_pipeline(...)["balances"]``): con duplicados, nulos y lecturas
    corruptas. El forecast del squad es ``tail(14).mean()`` de esas filas (omite nulos), igual que su función.
    Devuelve ``(hoy, backtest)``: el forecast a ``horizons[0]`` días desde el último día, y el error por cuenta × horizonte.
    """
    bal = panel.pivot(index="date", columns="account_id", values="balance").sort_index()
    T, w = len(bal), cfg.SQUAD_WINDOW
    rows_by_acc = {a: g.sort_values("date", kind="stable") for a, g in raw_naive.groupby("account_id")}

    def squad_raw(acc, day):
        return squad_raw_mean(rows_by_acc[acc], day, w)

    def median_level(t, a, h):
        f: OriginForecast = store[t]
        cum = np.cumsum(f.draw(n_draws, np.random.default_rng([cfg.BOOTSTRAP_SEED, t]))[:, :h, a], axis=1)[:, -1]
        return bal.iloc[t, a], cum

    names = {"sin_limpiar": "squad, sin limpiar", "limpio": "squad, limpio", "propuesto": "propuesto, limpio"}
    rows = []
    for h in horizons:
        for a, acc in enumerate(bal.columns):
            b = bal[acc].to_numpy()
            err = {k: [] for k in names}
            for t in range(start, T - h):
                _, cum = median_level(t, a, h)
                real = b[t + h]
                err["sin_limpiar"].append(squad_raw(acc, bal.index[t]) - real)
                err["limpio"].append(b[t - w + 1 : t + 1].mean() - real)
                err["propuesto"].append(b[t] + np.median(cum) - real)
            rows.append({"account_id": acc, "h": h,
                         **{f"mae_{k}": np.abs(v).mean() for k, v in err.items()},
                         **{f"max_err_{k}": np.abs(v).max() for k, v in err.items()},
                         **{f"sesgo_{k}": np.mean(v) for k, v in err.items()}})
    bt = pd.DataFrame(rows)
    for k in ("limpio", "propuesto"):
        bt[f"mae_{k}/sin_limpiar"] = bt[f"mae_{k}"] / bt["mae_sin_limpiar"]

    h0, t0 = horizons[0], T - 1
    today = []
    for a, acc in enumerate(bal.columns):
        last, cum = median_level(t0, a, h0)
        q5, q50, q95 = last + np.quantile(cum, [0.05, 0.5, 0.95])
        today.append({"account_id": acc, "ultimo_saldo": last, "squad_sin_limpiar": squad_raw(acc, bal.index[t0]),
                      "squad_limpio": bal[acc].to_numpy()[-w:].mean(), f"propuesto_p50_{h0}d": q50,
                      f"propuesto_p5_{h0}d": q5, f"propuesto_p95_{h0}d": q95})
    return pd.DataFrame(today).set_index("account_id"), bt


# ------------------------------------------------------------------- D6 / D7
def donor_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Saldo más reciente en moneda nativa y en USD (FX de config) y el ranking que usa el squad vs el real."""
    last = panel[panel["date"] == panel["date"].max()].set_index("account_id")
    out = last[["currency", "account_type", "balance"]].copy()
    out["balance_usd_eq"] = out["balance"] / out["currency"].map(cfg.FX_PER_USD)
    out["rank_nominal (squad)"] = out["balance"].rank(ascending=False).astype(int)
    out["rank_usd_eq"] = out["balance_usd_eq"].rank(ascending=False).astype(int)
    return out


def threshold_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Umbrales del squad: cobertura (4 de 6) y dónde caen respecto de la propia distribución de cada cuenta."""
    rows = []
    for acc, g in panel.groupby("account_id"):
        thr = cfg.SQUAD_THRESHOLD.get(acc)
        rows.append(
            {
                "account_id": acc, "tipo": g["account_type"].iloc[0], "moneda": g["currency"].iloc[0],
                "umbral_squad": thr, "saldo_mediano": g["balance"].median(),
                "umbral / mediana": (thr / g["balance"].median()) if thr else np.nan,
                "% de dias con saldo < umbral": float((g["balance"] < thr).mean() * 100) if thr else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("account_id")


# ------------------------------------------------------------------------ D10
def pre_post(raw_naive: pd.DataFrame, panel: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """La 'validación' del squad (promedio de saldos antes/después del 1-may, sumando monedas) y su versión por moneda."""
    cut = pd.Timestamp(cfg.SQUAD_CUTOVER)
    rows = []

    def add(label, df):
        pre, post = df.loc[df.date < cut, "balance"].mean(), df.loc[df.date >= cut, "balance"].mean()
        rows.append({"vista": label, "antes": pre, "despues": post, "cambio_%": (post / pre - 1) * 100})

    add("squad: COP+USD+MXN sumados, datos crudos", raw_naive)
    for ccy in ("COP", "USD", "MXN"):
        add(f"{ccy} solo, datos limpios", panel[panel.currency == ccy])
    return pd.DataFrame(rows)


def currency_weight_in_squad_mean(raw_naive: pd.DataFrame, accounts: pd.DataFrame) -> pd.Series:
    """Qué parte del 'promedio de saldos' del squad aporta cada moneda (por magnitud nominal)."""
    ccy = raw_naive["account_id"].map(accounts.set_index("account_id")["currency"])
    s = raw_naive["balance"].abs().groupby(ccy).sum()
    return s / s.sum()


# --------------------------------------------------------------- solvencia del piso
def system_surplus_by_k(panel: pd.DataFrame, ks=(20, 25, 28, 30)) -> pd.DataFrame:
    """Excedente agregado (USD eq) = Σ (saldo − K·egreso medio) sobre las cuentas, en la ventana del backtest."""
    from .policy import Book

    rows = []
    for k in ks:
        book = Book.from_panel(panel, cover_days={"operational": float(k), "reserve": float(k)})
        usd = 1.0 / book.fx
        sl = slice(cfg.MIN_TRAIN_DAYS, None)
        surplus = ((book.balance - book.floor) * usd)[sl]
        rows.append(
            {
                "K (días de egresos)": k,
                "excedente_agregado_min_USD": float(surplus.sum(axis=1).min()),
                "excedente_agregado_final_USD": float(surplus.sum(axis=1)[-1]),
                "dias-cuenta bajo el piso (sin intervenir)": int((book.balance[sl] < book.floor[sl]).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("K (días de egresos)")
