"""Figuras del caso. Cada una se guarda en reports/figures/ con ejes etiquetados y moneda indicada."""
from __future__ import annotations

import matplotlib
import matplotlib.dates

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

PALETTE = {"squad": "#C44E52", "squad2": "#DD8452", "none": "#8C8C8C", "quantile": "#4C72B0", "lp": "#2A9D8F", "clean": "#4C72B0"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name: str):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=130)
    return fig


def _fmt(ccy: str) -> str:
    return {"COP": "millones COP", "MXN": "millones MXN", "USD": "miles USD"}[ccy]


def _scale(ccy: str) -> float:
    return {"COP": 1e6, "MXN": 1e6, "USD": 1e3}[ccy]


# tinta y series de fig_cleaning: crudo = neutro, limpio = azul, lo corregido/afectado = naranja
_INK, _INK2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
_BEFORE, _AFTER, _FLAG = "#8a8984", "#2a78d6", "#eb6834"
CORRUPT_REL = 0.01   # una lectura del crudo es "corrupta" si difiere > 1% del valor reconciliado


def fig_cleaning(raw: pd.DataFrame, panel: pd.DataFrame):
    """Sin limpieza (lo que ve el pipeline del squad: coerce + dropna) vs. con limpieza (panel reconciliado).

    Una fila por cuenta, cada columna con su propia escala: en la izquierda la banda azul tenue es el rango del
    saldo limpio, para ver a qué distancia caen las lecturas corruptas (naranja). Los días sin dato son los que el
    squad pierde por fecha no parseable o balance nulo.
    """
    naive = raw.assign(d=pd.to_datetime(raw["date"], errors="coerce")).dropna(subset=["d"])
    accounts = sorted(panel["account_id"].unique())
    fig, axes = plt.subplots(len(accounts), 2, figsize=(13, 2.0 * len(accounts) + 1.4), sharex=True,
                             facecolor=_SURFACE, constrained_layout=True)
    for (ax_b, ax_a), acc in zip(axes, accounts):
        g = panel[panel["account_id"] == acc].sort_values("date")
        ccy, s = g["currency"].iloc[0], _scale(g["currency"].iloc[0])
        n = naive[naive["account_id"] == acc].sort_values("d").merge(
            g[["date", "balance"]].rename(columns={"balance": "clean"}), left_on="d", right_on="date", how="left")
        bad = (n["balance"] - n["clean"]).abs() > CORRUPT_REL * n["clean"].abs()
        sin_dato = len(set(g["date"]) - set(n.loc[n["balance"].notna(), "d"]))
        rep = g[g["balance_status"].str.startswith("repaired")]
        q = g[g["balance_status"] == "quarantine_bridged"]

        ax_b.axhspan(g["balance"].min() / s, g["balance"].max() / s, color=_AFTER, alpha=0.10, lw=0)
        ax_b.plot(n["d"], n["balance"] / s, color=_BEFORE, lw=1.3)
        ax_b.scatter(n.loc[bad, "d"], n.loc[bad, "balance"] / s, s=34, color=_FLAG, edgecolors=_SURFACE, linewidths=1.2, zorder=4)

        ax_a.plot(g["date"], g["balance"] / s, color=_AFTER, lw=1.6)
        for _, grp in q.groupby((q["date"].diff().dt.days != 1).cumsum()):   # ±1 día de margen solo para que se vea
            ax_a.axvspan(grp["date"].min() - pd.Timedelta(days=1), grp["date"].max() + pd.Timedelta(days=1), color=_FLAG, alpha=0.20, lw=0)
        ax_a.scatter(rep["date"], rep["balance"] / s, marker="D", s=36, color=_FLAG, edgecolors=_SURFACE, linewidths=1.2, zorder=4)

        for ax, title, stats in (
            (ax_b, f"{acc} · {g['account_type'].iloc[0]} — sin limpieza", f"{sin_dato} de {len(g)} días sin dato · {int(bad.sum())} lecturas corruptas"),
            (ax_a, f"{acc} — con limpieza", f"reparadas: {len(rep)} · días en cuarentena: {len(q)}"),
        ):
            ax.set_title(title, loc="left", fontsize=9.5, color=_INK)
            ax.text(1, 1.02, stats, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=_INK2)
            ax.set_facecolor(_SURFACE)
            ax.grid(axis="y", color=_GRID, lw=0.8); ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(_GRID)
            ax.tick_params(colors=_INK2, labelsize=8, length=0)
            ax.margins(y=0.08)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax_b.set_ylabel(_fmt(ccy), fontsize=8.5, color=_INK2)

    axes[-1, 0].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axes[-1, 0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    handles = [
        plt.Line2D([], [], color=_BEFORE, lw=1.6, label="Sin limpieza: pipeline del squad"),
        plt.Line2D([], [], color=_AFTER, lw=2, label="Con limpieza: panel reconciliado"),
        plt.Line2D([], [], marker="o", ls="", ms=7, color=_FLAG, mec=_SURFACE, label="Lectura corrupta / episodio en cuarentena / reparada"),
        plt.Rectangle((0, 0), 1, 1, color=_AFTER, alpha=0.10, label="Rango del saldo limpio (solo columna izquierda)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=_INK2)
    fig.suptitle("Saldo diario por cuenta, 2025 — sin limpieza vs. con limpieza (cada panel con su propia escala)",
                 fontsize=12, color=_INK, x=0.01, ha="left")
    return _save(fig, "01_cleaning")


def fig_trend(panel: pd.DataFrame, diff_table: pd.DataFrame, account_id: str, serie: str = "balance"):
    """STL nativo de statsmodels (observado / tendencia / estacional / residuo) de ``serie`` en ``account_id``,
    para VER la tendencia antes del ACF/PACF. Es el ``.plot()`` que trae ``STLForecastResults`` de fábrica:
    sin estilo custom, una figura por cuenta.
    """
    from statsmodels.tsa.seasonal import STL

    s = panel[panel["account_id"] == account_id].sort_values("date").set_index("date")[serie].astype(float).asfreq("D")
    res = STL(s, period=7, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(9, 6)
    d = int(diff_table.loc[(account_id, serie), "d"]) if (account_id, serie) in diff_table.index else 0
    fig.suptitle(f"{account_id} · {serie} ({'se diferencia' if d else 'se le resta esta tendencia'})")
    return _save(fig, f"09a_trend_{account_id}_{serie}")


def fig_acf_pacf(stationary: dict, diff_table: pd.DataFrame, which: str = "acf"):
    """ACF o PACF de la serie estacionaria de cada cuenta × serie (saldo, inflow, outflow).

    ``stationary`` y ``diff_table`` salen de ``seasonality.acf_pacf_report``. Naranja = rezagos 7/14/21/28; la banda
    gris es la del 95% de un ruido blanco (±1,96/sqrt(n)) — todo lo que quede dentro es indistinguible de ruido. Los
    rezagos estacionales que sí sobresalen de la banda llevan su valor escrito encima.
    """
    from statsmodels.tsa.stattools import acf, pacf

    accounts = sorted({a for a, _ in stationary})
    series = sorted({c for _, c in stationary}, key=["balance", "inflow", "outflow"].index)

    # una pasada previa para conocer todos los valores y fijar un ylim común que no recorte ni sobre-amplifique
    cache, peak = {}, 0.0
    for acc in accounts:
        for col in series:
            x = stationary[(acc, col)]
            vals = (acf(x, nlags=cfg.ACF_NLAGS, fft=True) if which == "acf" else pacf(x, nlags=cfg.ACF_NLAGS, method="ywm"))[1:]
            band = 1.96 / np.sqrt(len(x))
            cache[(acc, col)] = (vals, band)
            peak = max(peak, np.abs(vals).max())
    ylim = max(0.3, min(1.0, peak * 1.25))

    fig, axes = plt.subplots(len(accounts), len(series), figsize=(13, 1.9 * len(accounts) + 1.2), sharex=True, sharey=True,
                             facecolor=_SURFACE, constrained_layout=True)
    for i, acc in enumerate(accounts):
        for j, col in enumerate(series):
            ax = axes[i, j]
            vals, band = cache[(acc, col)]
            lags = np.arange(1, len(vals) + 1)
            sig = np.abs(vals) > band

            ax.axhspan(-band, band, color=_GRID, alpha=0.55, zorder=1, lw=0)
            ax.bar(lags, vals, width=0.65, color=[_FLAG if l in cfg.SEASONAL_LAGS else _AFTER for l in lags], zorder=3)
            ax.axhline(0, color=_INK2, lw=0.8, zorder=2)
            for l, v in zip(lags, vals):
                if l in cfg.SEASONAL_LAGS and abs(v) > band:
                    ax.annotate(f"{v:.2f}", (l, v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                                ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, color=_FLAG, fontweight="bold", zorder=4)
            ax.set_title(f"{acc} · {col} ({'diferencia' if diff_table.loc[(acc, col), 'd'] else 'sin tendencia'})",
                         loc="left", fontsize=9, color=_INK)
            ax.text(1, 1.02, f"{int(sig.sum())}/{len(vals)} fuera de banda", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=_INK2)
            ax.set_facecolor(_SURFACE)
            ax.set_ylim(-ylim, ylim)
            ax.set_xticks([7, 14, 21, 28])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(_GRID)
            ax.tick_params(colors=_INK2, labelsize=8, length=0)
            ax.grid(axis="y", color=_GRID, lw=0.6, zorder=0); ax.grid(axis="x", visible=False)
        axes[i, 0].set_ylabel(which.upper(), fontsize=8.5, color=_INK2)
    axes[-1, len(series) // 2].set_xlabel("rezago (días)", fontsize=8.5, color=_INK2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_AFTER, label="rezago"),
        plt.Rectangle((0, 0), 1, 1, color=_FLAG, label="rezago estacional (7, 14, 21, 28)"),
        plt.Rectangle((0, 0), 1, 1, color=_GRID, alpha=0.55, label="banda 95% de un ruido blanco"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=_INK2)
    fig.suptitle(f"{which.upper()} de la serie estacionaria: ¿hay estructura y patrón semanal?", fontsize=12, color=_INK, x=0.01, ha="left")
    return _save(fig, "09_acf" if which == "acf" else "10_pacf")


def fig_seasonality(panel: pd.DataFrame, serie: str):
    """Perfil de estacionalidad de UNA serie (saldo, inflow u outflow): cuentas en filas, ciclos en columnas.

    Cada panel muestra la desviación % media respecto de la tendencia (media móvil centrada de 28 días) por posición del ciclo,
    con la banda ±1,96 errores estándar y el p-valor de Kruskal-Wallis. Si la banda incluye el 0 casi en todas las posiciones,
    no hay patrón distinguible del ruido. Sirve para VER la estacionalidad antes de probar estacionariedad (ADF/KPSS).
    """
    from . import seasonality

    accounts = sorted(panel["account_id"].unique())
    cycles = seasonality.CYCLES
    labels = {"semanal": ["L", "M", "X", "J", "V", "S", "D"]}
    fig, axes = plt.subplots(len(accounts), len(cycles), figsize=(13, 1.9 * len(accounts) + 1.2), sharey="row",
                             facecolor=_SURFACE, constrained_layout=True)
    for i, acc in enumerate(accounts):
        s = panel[panel["account_id"] == acc].sort_values("date").set_index("date")[serie]
        for j, cycle in enumerate(cycles):
            ax = axes[i, j]
            tab, p = seasonality.cycle_profile(s, cycle)
            x = tab["pos"].to_numpy()
            ax.axhline(0, color=_INK2, lw=0.8, zorder=1)
            ax.fill_between(x, tab["media"] - 1.96 * tab["se"], tab["media"] + 1.96 * tab["se"], color=_BEFORE, alpha=0.25, lw=0, zorder=2)
            ax.plot(x, tab["media"], color=_AFTER, lw=1.6, marker="o", ms=3.5, zorder=3)
            if cycle == "semanal":
                ax.set_xticks(range(7)); ax.set_xticklabels(labels["semanal"])
            elif cycle == "quincenal":
                ax.set_xticks([1, 5, 10, 15])
            else:
                ax.set_xticks([1, 8, 15, 22, 29])
            sig = np.isfinite(p) and p < 0.01
            ax.set_title(f"{acc} · {cycle}", loc="left", fontsize=9, color=_INK)
            ax.text(1, 1.02, f"p = {p:.2g}" if np.isfinite(p) else "p = n/d", transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    color=_FLAG if sig else _INK2, fontweight="bold" if sig else "normal")
            ax.set_facecolor(_SURFACE)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(_GRID)
            ax.tick_params(colors=_INK2, labelsize=8, length=0)
            ax.grid(axis="y", color=_GRID, lw=0.6, zorder=0); ax.grid(axis="x", visible=False)
        axes[i, 0].set_ylabel("% vs tendencia", fontsize=8.5, color=_INK2)
    handles = [plt.Line2D([], [], color=_AFTER, lw=1.6, marker="o", ms=4, label="perfil medio (% vs tendencia de 28 días)"),
               plt.Rectangle((0, 0), 1, 1, color=_BEFORE, alpha=0.25, label="banda ±1,96 errores estándar"),
               plt.Line2D([], [], color=_FLAG, lw=0, marker="s", ms=6, label="p < 0,01 (Kruskal-Wallis)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=_INK2)
    fig.suptitle(f"Estacionalidad de {serie}: semanal, quincenal (día dentro de la quincena) y mensual (día del mes)", fontsize=12, color=_INK, x=0.01, ha="left")
    return _save(fig, f"11_estacionalidad_{serie}")


def fig_squad_forecast(actual: pd.DataFrame, account: str = "ACC-004", h: int = 7):
    """Pronóstico del squad (media 14d del nivel) vs saldo real a h días vs. 'último saldo'."""
    a = actual[actual["account_id"] == account]
    ccy = a["currency"].iloc[0]
    s = _scale(ccy)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(a["date"], a["realized"] / s, color="k", lw=1.6, label=f"saldo real a {h} días")
    ax.plot(a["date"], a["squad"] / s, color=PALETTE["squad"], lw=1.4, label="squad: media 14d del nivel")
    ax.plot(a["date"], a["last"] / s, color=PALETTE["quantile"], lw=1.0, ls="--", label="referencia: último saldo")
    ax.set_ylabel(_fmt(ccy)); ax.set_title(f"{account}: el pronóstico del squad va rezagado respecto del saldo real (horizonte {h}d)")
    ax.legend()
    return _save(fig, "02_squad_forecast")


def fig_lag_fee(transfers: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ct = pd.crosstab(transfers["kind"], transfers["lag_days"])
    ct.T.plot.bar(ax=axes[0], color=[PALETTE["quantile"], PALETTE["squad"]], rot=0)
    axes[0].set_xlabel("lag de liquidación (días)"); axes[0].set_ylabel("transferencias"); axes[0].set_title("Lag por tipo de transferencia")
    for kind, col in (("same_ccy", PALETTE["quantile"]), ("cross_ccy", PALETTE["squad"])):
        axes[1].hist(transfers.loc[transfers["kind"] == kind, "fee_rate"] * 100, bins=20, alpha=0.7, color=col, label=kind)
    axes[1].set_xlabel("fee (% del monto)"); axes[1].set_ylabel("transferencias"); axes[1].set_title("Fee por tipo"); axes[1].legend()
    fig.tight_layout()
    return _save(fig, "03_lag_fee")


def fig_flow_structure(panel: pd.DataFrame):
    """Estructura de los flujos: día de semana (bruto) y autocorrelación del neto."""
    p = panel.assign(dow=panel["date"].dt.dayofweek)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for acc, g in p.groupby("account_id"):
        prof = g.groupby("dow")["outflow"].mean() / g["outflow"].mean()
        axes[0].plot(["L", "M", "X", "J", "V", "S", "D"], prof.values, marker="o", label=acc)
        prof_n = g.groupby("dow")["net_flow"].mean() / g["net_flow"].std()
        axes[1].plot(["L", "M", "X", "J", "V", "S", "D"], prof_n.values, marker="o", label=acc)
    axes[0].set_title("Egreso bruto por día de semana\n(índice, media = 1): el fin de semana cae ~50%"); axes[0].set_ylabel("índice (adimensional)")
    axes[1].set_title("Flujo neto medio por día de semana\n(en desv. estándar): ~ruido"); axes[1].set_ylabel("media / desv. estándar"); axes[1].axhline(0, color="k", lw=0.6)
    lags = np.arange(1, 15)
    for acc, g in p.groupby("account_id"):
        x = g["net_flow"].to_numpy()
        axes[2].plot(lags, [pd.Series(x).autocorr(l) for l in lags], marker=".", label=acc)
    axes[2].axhline(0, color="k", lw=0.6); axes[2].set_title("Autocorrelación del flujo neto"); axes[2].set_xlabel("rezago (días)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, "04_flow_structure")


def fig_model_skill(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    summary[["mae_cum_rel", "pinball_cum_rel"]].rename(columns={"mae_cum_rel": "MAE (acumulado)", "pinball_cum_rel": "pinball (acumulado)"}).plot.bar(ax=axes[0], rot=15, color=[PALETTE["none"], PALETTE["quantile"]])
    axes[0].axhline(1, color="k", lw=0.8, ls="--"); axes[0].set_ylabel("métrica / métrica del baseline"); axes[0].set_title("Error relativo al naive estacional (<1 = mejor)")
    (summary["cover_cum"] * 100).plot.bar(ax=axes[1], rot=15, color=PALETTE["lp"])
    axes[1].axhline(90, color="k", lw=0.8, ls="--"); axes[1].set_ylim(70, 100); axes[1].set_ylabel("% real dentro de p5–p95"); axes[1].set_title("Cobertura empírica del intervalo (nominal 90%)")
    fig.tight_layout()
    return _save(fig, "05_model_skill")


def fig_fan(book, store, t: int, account: str, horizon: int = cfg.DECISION_HORIZON, n: int = 2000):
    """Saldo proyectado p5–p95 desde el origen ``t`` vs el saldo realizado y el umbral fijo del squad (D7), donde exista."""
    a = book.accounts.index(account)
    f = store[t]
    paths = f.draw(n, np.random.default_rng(1))[:, :, a]
    proj = book.balance[t, a] + np.cumsum(paths, axis=1)
    q = np.quantile(proj, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    s, ccy = _scale(book.ccy[a]), book.ccy[a]
    fig, ax = plt.subplots(figsize=(10, 4))
    past = slice(max(t - 30, 0), t + 1)
    ax.plot(book.dates[past], book.balance[past, a] / s, color="k", lw=1.6, label="saldo observado")
    real = slice(t, min(t + horizon + 1, len(book.dates)))
    ax.plot(book.dates[real], book.balance[real, a] / s, color="k", lw=1.6, ls=":", label="saldo realizado (fuera de muestra)")
    d = pd.DatetimeIndex([book.dates[t]]).append(f.dates)
    ax.fill_between(f.dates, q[0] / s, q[4] / s, color=PALETTE["quantile"], alpha=0.18, label="p5–p95")
    ax.fill_between(f.dates, q[1] / s, q[3] / s, color=PALETTE["quantile"], alpha=0.30, label="p25–p75")
    ax.plot(f.dates, q[2] / s, color=PALETTE["quantile"], label="mediana")
    thr = cfg.SQUAD_THRESHOLD.get(account)
    if thr is not None:
        ax.axhline(thr / s, color=PALETTE["squad"], ls="--", lw=1.2, label="umbral fijo del squad (D7)")
    ax.set_ylabel(_fmt(ccy)); ax.set_title(f"{account}: saldo proyectado a {horizon} días desde {book.dates[t].date()} (distribución, no punto)")
    ax.legend(fontsize=8)
    return _save(fig, "06_fan_chart")


def fig_fan_models(book, flows: pd.DataFrame, t: int, account: str, models: tuple[str, ...] | None = None,
                    horizon: int = cfg.DECISION_HORIZON, n: int = 2000):
    """Como ``fig_fan``, pero un panel por modelo de F3 (mismo origen ``t``, misma cuenta): compara sus bandas
    p5–p95 lado a lado en vez de solo el modelo que usa ``store`` (``empirical``, el elegido en 2.3).

    Cada panel es un ``forecast.fit_origin`` distinto sobre los mismos datos hasta ``t`` — nada de lookahead.
    La línea horizontal es el umbral fijo del squad (D7, ``cfg.SQUAD_THRESHOLD``), donde exista para la cuenta;
    no el piso dinámico (K días de egreso), que aún no se ha justificado en esta sección.
    """
    from . import forecast as Fc

    models = tuple(models or Fc.MODEL_ORDER)
    a = book.accounts.index(account)
    s, ccy = _scale(book.ccy[a]), book.ccy[a]
    past = slice(max(t - 30, 0), t + 1)
    real = slice(t, min(t + horizon + 1, len(book.dates)))
    thr = cfg.SQUAD_THRESHOLD.get(account)

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.3), sharex=True, sharey=True,
                             facecolor=_SURFACE, constrained_layout=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        f = Fc.fit_origin(flows, t, model, H=horizon)
        paths = f.draw(n, np.random.default_rng(1))[:, :, a]
        proj = book.balance[t, a] + np.cumsum(paths, axis=1)
        q = np.quantile(proj, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

        ax.plot(book.dates[past], book.balance[past, a] / s, color=_INK, lw=1.3, zorder=4)
        ax.plot(book.dates[real], book.balance[real, a] / s, color=_INK, lw=1.3, ls=":", zorder=4)
        ax.fill_between(f.dates, q[0] / s, q[4] / s, color=PALETTE["quantile"], alpha=0.18, lw=0, zorder=2)
        ax.fill_between(f.dates, q[1] / s, q[3] / s, color=PALETTE["quantile"], alpha=0.32, lw=0, zorder=2)
        ax.plot(f.dates, q[2] / s, color=PALETTE["quantile"], lw=1.6, zorder=3)

        ax.set_title(model, loc="left", fontsize=10, color=_INK)
        if thr is not None:
            ax.axhline(thr / s, color=PALETTE["squad"], ls="--", lw=1.1, zorder=3)
            below = float((proj[:, -1] < thr).mean())
            ax.text(1, 1.02, f"P(< umbral squad) a {horizon}d = {below:.0%}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color=_INK2)
        ax.set_facecolor(_SURFACE)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_INK2, labelsize=7.5, length=0, rotation=30)
        ax.grid(axis="y", color=_GRID, lw=0.6, zorder=0); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(_fmt(ccy), fontsize=8.5, color=_INK2)
    handles = [
        plt.Line2D([], [], color=_INK, lw=1.3, label="saldo observado / realizado (punteado)"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["quantile"], alpha=0.18, label="p5–p95"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["quantile"], alpha=0.32, label="p25–p75"),
        plt.Line2D([], [], color=PALETTE["quantile"], lw=1.6, label="mediana"),
    ]
    if thr is not None:
        handles.append(plt.Line2D([], [], color=PALETTE["squad"], ls="--", lw=1.1, label="umbral fijo del squad (D7)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=8, labelcolor=_INK2)
    fig.suptitle(f"{account}: saldo proyectado a {horizon} días desde {book.dates[t].date()} — los {len(models)} modelos de F3",
                 fontsize=12, color=_INK, x=0.01, ha="left")
    return _save(fig, f"06b_fan_models_{account}")


def fig_frontier(totals: pd.DataFrame, highlight_lp: float = cfg.DEFICIT_PENALTY_PER_DAY):
    """Frontera costo–riesgo: costo total (USD, fees + costo fijo) vs déficit acumulado (USD·día)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    style = {"0": ("Sin transferencias (contrafactual)", PALETTE["none"], "s"), "A": ("Squad (A / A2)", PALETTE["squad"], "X"),
             "B": ("Política de cuantil (B)", PALETTE["quantile"], "o"), "C": ("MILP (C)", PALETTE["lp"], "D")}
    done = set()
    for _, r in totals.iterrows():
        lbl, col, mk = style[r["familia"]]
        ax.scatter(r["costo_total_usd"], r["deficit_usd_dias"], color=col, marker=mk, s=95, zorder=4, label=None if lbl in done else lbl, edgecolor="k", lw=0.5)
        done.add(lbl)
        ax.annotate(r["politica"].split(": ")[1] if ": " in r["politica"] else "sin", (r["costo_total_usd"], r["deficit_usd_dias"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7)
    for fam, col in (("B", PALETTE["quantile"]), ("C", PALETTE["lp"])):
        sub = totals[totals["familia"] == fam].sort_values("costo_total_usd")
        ax.plot(sub["costo_total_usd"], sub["deficit_usd_dias"], color=col, lw=1, alpha=0.6)
    ax.set_yscale("symlog", linthresh=1e4)
    ax.set_xlabel("Costo total (USD equivalente) = fees + costo fijo por operación")
    ax.set_ylabel("Déficit acumulado bajo el piso (USD·día, equivalente a FX de config)")
    ax.set_title("Frontera costo–riesgo: abajo-izquierda es mejor")
    ax.legend(loc="upper right")
    return _save(fig, "07_frontier")


def fig_policy_paths(results: dict, book, accounts=("ACC-004", "ACC-002", "ACC-005", "ACC-003"), names=None):
    names = names or list(results)
    colors = {"0": PALETTE["none"], "A": PALETTE["squad"], "B": PALETTE["quantile"], "C": PALETTE["lp"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    t0 = next(iter(results.values())).t0
    for ax, acc in zip(axes.ravel(), accounts):
        a = book.accounts.index(acc)
        s = _scale(book.ccy[a])
        for n in names:
            r = results[n]
            ax.plot(book.dates[t0:], r.balance[t0:, a] / s, color=colors[n[0]], lw=1.3, alpha=0.9, label=n)
        ax.plot(book.dates[t0:], book.floor[t0:, a] / s, color="k", ls="--", lw=1, label="piso")
        ax.set_title(f"{acc} ({book.acc_type[a]})"); ax.set_ylabel(_fmt(book.ccy[a]))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Saldo simulado por política vs. piso operativo", y=1.0)
    fig.tight_layout()
    return _save(fig, "08_policy_paths")
