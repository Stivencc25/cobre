"""F2 — Features y reconciliación."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Día de semana, fin de semana, fin de mes y flujo neto acumulado."""
    p = panel.sort_values(["account_id", "date"]).copy()
    p["dow"] = p["date"].dt.dayofweek
    p["is_weekend"] = p["dow"] >= 5
    p["is_month_end"] = p["date"].dt.is_month_end
    p["cum_net_flow"] = p.groupby("account_id")["net_flow"].cumsum()
    return p


def add_in_transit(panel: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Monto en tránsito al cierre de cada día: solicitado (<= t) y aún no liquidado (> t).

    ``in_transit_out_native``: sale de la cuenta origen en su moneda.
    ``in_transit_in_native_at_cfg_fx``: llegará a la cuenta destino, convertido a FX de config.
    """
    fx = cfg.FX_PER_USD
    p = panel.copy()
    p["in_transit_out_native"] = 0.0
    p["in_transit_in_native_at_cfg_fx"] = 0.0
    for tr in transfers.itertuples():
        recv = tr.amount * fx[tr.to_ccy] / fx[tr.from_ccy]
        mask_days = (p["date"] >= tr.date_requested) & (p["date"] < tr.date_settled)
        p.loc[mask_days & (p["account_id"] == tr.from_account), "in_transit_out_native"] += tr.amount
        p.loc[mask_days & (p["account_id"] == tr.to_account), "in_transit_in_native_at_cfg_fx"] += recv
    return p


def reconciliation_report(panel: pd.DataFrame, transfers: pd.DataFrame, tol: float = cfg.RECONCILIATION_TOL):
    """Residuo de la identidad por cuenta y prueba de si el log de transferencias lo explica.

    El spec (§3.7b) anticipa residuos en días con transferencia liquidada. Aquí se PRUEBA esa
    hipótesis: si el log estuviera reflejado en los saldos, esos días tendrían residuo
    material igual al monto de la transferencia.
    Devuelve ``(resumen_por_cuenta, tabla_prueba_transferencias, residuos_materiales)``.
    """
    p = panel.sort_values(["account_id", "date"]).copy()
    p["residual"] = p["balance"] - (p.groupby("account_id")["balance"].shift(1) + p["inflow"] - p["outflow"])
    settle = pd.concat(
        [
            transfers[["from_account", "date_settled", "transfer_id"]].rename(columns={"from_account": "account_id"}),
            transfers[["to_account", "date_settled", "transfer_id"]].rename(columns={"to_account": "account_id"}),
        ]
    ).rename(columns={"date_settled": "date"})
    ids = settle.groupby(["account_id", "date"])["transfer_id"].agg(lambda s: ",".join(sorted(s))).rename("transfer_ids")
    p = p.merge(ids, left_on=["account_id", "date"], right_index=True, how="left")
    p["settlement_day"] = p["transfer_ids"].notna()
    ev = p.dropna(subset=["residual"]).copy()
    ev["material"] = ev["residual"].abs() > tol

    by_account = ev.groupby("account_id").agg(
        dias_evaluados=("residual", "size"),
        residuo_abs_max=("residual", lambda s: s.abs().max()),
        dias_con_residuo_material=("material", "sum"),
        dias_con_settlement=("settlement_day", "sum"),
    )
    test = pd.DataFrame(
        {
            "días cuenta con transferencia liquidada": [int(ev["settlement_day"].sum())],
            "…con residuo material": [int((ev["settlement_day"] & ev["material"]).sum())],
            "días sin transferencia con residuo material": [int((~ev["settlement_day"] & ev["material"]).sum())],
        }
    )
    return by_account, test, ev[ev["material"]]


def transfer_vs_flow_scale(panel: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Compara el tamaño de las transferencias con el flujo diario máximo de su moneda.

    Si una transferencia USD mediana supera el flujo diario máximo de las cuentas USD, no puede
    estar 'escondida' dentro de inflow/outflow: es evidencia independiente de que el log no está
    en los saldos.
    """
    rows = []
    for ccy, g in transfers.groupby("from_ccy"):
        accs = panel.loc[panel["currency"] == ccy]
        rows.append(
            {
                "moneda": ccy,
                "transferencia_mediana": g["amount"].median(),
                "flujo_diario_max (in/out)": max(accs["inflow"].max(), accs["outflow"].max()),
                "mediana / flujo_max": g["amount"].median() / max(accs["inflow"].max(), accs["outflow"].max()),
            }
        )
    return pd.DataFrame(rows).set_index("moneda")
