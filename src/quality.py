"""F1 — Reporte de calidad: cuántas filas afecta cada problema y qué se hizo con ellas."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as cfg

CORRUPT_REL = 0.01   # una lectura del crudo es "corrupta" si difiere > 1% del valor reconciliado (igual que plots.fig_cleaning)


def naive_parse_loss(raw: pd.DataFrame) -> tuple[int, int]:
    """Filas que el pipeline del squad (`to_datetime(errors='coerce')` + `dropna`) pierde en silencio."""
    naive = pd.to_datetime(raw["date"], errors="coerce")
    return int(naive.isna().sum()), len(raw)


def date_format_counts(raw: pd.DataFrame) -> pd.Series:
    def label(v: str) -> str:
        for (pattern, fmt) in cfg.DATE_FORMATS:
            if re.fullmatch(pattern, v):
                return fmt
        return "desconocido"

    return raw["date"].map(label).value_counts()


def identity_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Residuo  balance[t] - (balance[t-1] + inflow - outflow)  por cuenta."""
    p = panel.sort_values(["account_id", "date"]).copy()
    p["residual"] = p["balance"] - (p.groupby("account_id")["balance"].shift(1) + p["inflow"] - p["outflow"])
    return p.dropna(subset=["residual"])


def quality_report(
    raw: pd.DataFrame,
    ledger: pd.DataFrame,
    panel: pd.DataFrame,
    transfers_raw: pd.DataFrame,
    transfers_excluded: pd.DataFrame,
) -> pd.DataFrame:
    """Tabla: problema -> filas/eventos afectados -> tratamiento. Incluye filas entrantes y salientes."""
    lost, total = naive_parse_loss(raw)
    labels = raw["currency"].nunique()
    dups = ledger[ledger["n_rows"] == 2]
    conflicting = dups[dups["n_candidates"] == 2]
    exact = dups[dups["n_candidates"] != 2]
    status = ledger["balance_status"].value_counts()
    n_neg_raw = int((raw["balance"] < 0).sum())
    audit = identity_audit(panel)

    rows = [
        ("Filas entrantes (crudo)", total, "—"),
        ("P1 fechas que el parser del squad pierde", lost, "cascada de 3 formatos: 0 no parseadas"),
        ("P2 etiquetas de moneda (deben ser 3)", labels, "normalizadas a ISO-4217 y validadas vs accounts.csv"),
        ("P2 filas con etiqueta no canónica", int((~raw["currency"].isin(["COP", "USD", "MXN"])).sum()), "→ moneda de la cuenta"),
        ("P3 pares (fecha, cuenta) duplicados con balance distinto", len(conflicting), "se conserva la fila que cuadra con la identidad"),
        ("P3 pares duplicados exactos/ambos nulos", len(exact), "se colapsan; balance por identidad"),
        ("P4 nulos en balance", int(raw["balance"].isna().sum()), "reconstruidos por identidad"),
        ("P4 nulos en inflow", int(raw["inflow"].isna().sum()), "recuperados por identidad o imputados (marca)"),
        ("P4 nulos en outflow", int(raw["outflow"].isna().sum()), "recuperados por identidad o imputados (marca)"),
        ("P5 balances negativos en el crudo", n_neg_raw, "todos sign-flip exacto (-1x) → reparados"),
        ("P5 reparados: signo invertido", int(status.get("repaired_sign_flip", 0)), "valor = lo que dice la identidad"),
        ("P5 reparados: decimal corrido (10x)", int(status.get("repaired_decimal_shift_x10", 0)), "valor = lo que dice la identidad"),
        ("P6 días-cuenta en cuarentena (episodios sin explicación)", int(status.get("unreconciled_episode", 0)), "puente lineal, marcado, reportado"),
        ("P6 lecturas plausibles pero no verificables", int(status.get("observed_unverified", 0)), "se aceptan con marca"),
        ("P7 residuo máx. de la identidad en el panel final", float(audit["residual"].abs().max()), "≤ 1 unidad en el 100% de los días"),
        ("Transferencias: duplicados exactos (mismo todo salvo id)", len(transfers_excluded), "excluidas y reportadas"),
        ("Filas salientes (panel cuenta×día)", len(panel), f"{cfg.EXPECTED_ACCOUNTS} cuentas × {cfg.EXPECTED_DAYS} días, 0 nulos"),
    ]
    return pd.DataFrame(rows, columns=["problema", "afectados", "tratamiento"]).astype({"afectados": object})


def cleaning_diff(raw: pd.DataFrame, ledger: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Día a día, solo donde el pipeline cambió algo: qué traía el crudo y qué queda en el panel.

    ``lo_ve_el_squad`` = el parser del squad (coerce + dropna) conserva la fecha de esa fila; si es False, el squad
    ni siquiera ve la lectura. ``desvio_*`` = cuánto se aparta del valor limpio la peor lectura del crudo.
    ``raw`` debe traer ``date_parsed`` (lo agrega ``ingest.run_pipeline``).
    """
    iso = pd.to_datetime(raw["date"], errors="coerce").notna()
    seen = raw.assign(_iso=iso).groupby(["account_id", "date_parsed"])["_iso"].any().rename("lo_ve_el_squad")
    clean = panel[["account_id", "date", "currency", "balance", "inflow", "outflow", "balance_status", "flow_status"]].rename(
        columns={"balance": "balance_limpio", "inflow": "inflow_limpio", "outflow": "outflow_limpio",
                 "balance_status": "estado_balance", "flow_status": "estado_flujo"})
    d = ledger[["account_id", "date", "n_candidates", "cand_1", "cand_2", "inflow_raw", "outflow_raw"]].merge(
        clean, on=["account_id", "date"])
    d = d.join(seen, on=["account_id", "date"]).assign(lo_ve_el_squad=lambda x: x["lo_ve_el_squad"].fillna(False))

    d["problema"] = np.select(
        [d["estado_balance"] == "repaired_sign_flip", d["estado_balance"] == "repaired_decimal_shift_x10",
         d["estado_balance"] == "quarantine_bridged", d["n_candidates"] == 2, d["n_candidates"] == 0,
         d["estado_flujo"] != "observed"],
        ["signo invertido", "decimal corrido x10", "cuarentena (puente lineal)", "duplicado con balance distinto",
         "balance nulo -> imputado", "flujo nulo/rehecho"],
        default="",
    )
    d = d[d["problema"] != ""].copy()
    gap = pd.concat([(d["cand_1"] - d["balance_limpio"]).abs(), (d["cand_2"] - d["balance_limpio"]).abs()], axis=1).max(axis=1)
    d["desvio_balance"] = gap
    d["desvio_pct"] = gap / d["balance_limpio"].abs() * 100
    d = d.rename(columns={"cand_1": "balance_crudo_1", "cand_2": "balance_crudo_2",
                          "inflow_raw": "inflow_crudo", "outflow_raw": "outflow_crudo"})
    cols = ["account_id", "date", "currency", "problema", "lo_ve_el_squad", "balance_crudo_1", "balance_crudo_2",
            "balance_limpio", "desvio_balance", "desvio_pct", "inflow_crudo", "inflow_limpio", "outflow_crudo",
            "outflow_limpio", "estado_balance", "estado_flujo"]
    return d[cols].sort_values(["account_id", "date"]).reset_index(drop=True)


def cleaning_summary(raw: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Comparativo por cuenta: lo que ve el squad (coerce + dropna, sin limpiar) vs. el panel limpio.

    Índice ``(account_id, métrica)``; columnas ``antes``, ``después`` y ``cambio_%`` (solo para magnitudes de saldo).
    Los conteos van como enteros y los saldos en la moneda nativa de la cuenta.
    """
    naive = raw.assign(d=pd.to_datetime(raw["date"], errors="coerce")).dropna(subset=["d"]).sort_values("d")
    rows = []
    for acc, g in panel.sort_values("date").groupby("account_id"):
        n = naive[naive["account_id"] == acc].merge(
            g[["date", "balance"]].rename(columns={"balance": "clean"}), left_on="d", right_on="date", how="left")
        corrupt = int(((n["balance"] - n["clean"]).abs() > CORRUPT_REL * n["clean"].abs()).sum())
        con_dato = int(n.loc[n["balance"].notna(), "d"].nunique())
        b, a = n["balance"], g["balance"]
        fc_before, fc_after = b.tail(cfg.SQUAD_WINDOW).mean(), a.tail(cfg.SQUAD_WINDOW).mean()

        def add(metric, before, after, pct=False):
            change = (after - before) / abs(before) * 100 if pct and before else np.nan   # |antes|: el mínimo crudo puede ser negativo
            rows.append((acc, metric, before, after, change))

        add("filas", len(n), len(g))
        add("días con balance", con_dato, len(g))
        add("días sin dato (fecha perdida o balance nulo)", len(g) - con_dato, 0)
        add("fechas duplicadas", int(n["d"].duplicated().sum()), 0)
        add("balances nulos", int(b.isna().sum()), 0)
        add("balances negativos", int((b < 0).sum()), int((a < 0).sum()))
        add(f"lecturas corruptas (> {CORRUPT_REL:.0%} del valor limpio)", corrupt, 0)
        add("días marcados no confiables (reliable = False)", np.nan, int((~g["reliable"]).sum()))
        add("saldo mínimo", b.min(), a.min(), pct=True)
        add("saldo máximo", b.max(), a.max(), pct=True)
        add("saldo medio", b.mean(), a.mean(), pct=True)
        add(f"forecast del squad hoy (media de los últimos {cfg.SQUAD_WINDOW})", fc_before, fc_after, pct=True)
    out = pd.DataFrame(rows, columns=["account_id", "métrica", "antes", "después", "cambio_%"], dtype=object)
    return out.set_index(["account_id", "métrica"])   # dtype=object: los conteos se muestran como enteros, los saldos como decimales

