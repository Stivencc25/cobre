"""F1 — Ingesta y corrección de datos.

Principios (ver SPEC §3, §8):
* Nada se descarta en silencio: toda corrección queda contada en el reporte de calidad.
* La identidad contable  balance[t] = balance[t-1] + inflow[t] - outflow[t]  es la llave de
  respuesta de los datos: elige el duplicado correcto, reconstruye nulos exactamente y
  detecta (y repara solo si hay firma exacta) lecturas corruptas.
* Lo que la identidad NO puede resolver se pone en cuarentena, se marca y se reporta.

Uso:  python -m src.ingest   -> regenera data/processed/
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as cfg

# --------------------------------------------------------------------------- carga


def load_accounts() -> pd.DataFrame:
    return pd.read_csv(cfg.ACCOUNTS_FILE)


def load_raw_balances() -> pd.DataFrame:
    """Lee el CSV crudo con la fecha como texto (el formato es mixto)."""
    return pd.read_csv(cfg.BALANCES_FILE, dtype={"date": str, "account_id": str, "currency": str})


def load_raw_transfers() -> pd.DataFrame:
    return pd.read_csv(cfg.TRANSFERS_FILE, parse_dates=["date_requested", "date_settled"])


# ---------------------------------------------------------------------- fechas


def date_format_of(value: str) -> str:
    """Devuelve el strptime que corresponde al patrón; falla si no reconoce ninguno."""
    for pattern, fmt in cfg.DATE_FORMATS:
        if re.fullmatch(pattern, value):
            return fmt
    raise ValueError(f"formato de fecha no reconocido: {value!r}")


def parse_mixed_dates(dates: pd.Series) -> pd.Series:
    """Parser en cascada sobre los 3 formatos presentes en el crudo.

    Supuesto verificado: el separador desambigua el formato (``/`` = DD/MM/YYYY, ``-`` con
    año al final = MM-DD-YYYY, ISO = YYYY-MM-DD), sin excepciones en las 1.668 filas.
    Falla ruidosamente si alguna fecha no encaja: nunca produce NaT en silencio.
    """
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum())} fechas nulas en el crudo")
    parsed = pd.Series(
        [pd.to_datetime(v, format=date_format_of(v)) for v in dates], index=dates.index, dtype="datetime64[ns]"
    )
    if parsed.isna().any():  # defensa: no debería ocurrir
        raise ValueError("quedaron fechas sin parsear")
    return parsed


# --------------------------------------------------------------------- moneda


def normalize_currency(labels: pd.Series, account_ids: pd.Series, accounts: pd.DataFrame) -> pd.Series:
    """Normaliza a ISO-4217 y valida contra ``accounts.csv`` (fuente de verdad).

    Falla si una etiqueta no está mapeada o si alguna fila contradice la moneda de su cuenta.
    """
    norm = labels.str.strip().str.lower().map(cfg.CURRENCY_LABEL_MAP)
    if norm.isna().any():
        raise ValueError(f"etiquetas de moneda sin mapear: {sorted(labels[norm.isna()].unique())}")
    truth = account_ids.map(accounts.set_index("account_id")["currency"])
    if (norm != truth).any():
        raise ValueError(f"{int((norm != truth).sum())} filas con moneda distinta a la de su cuenta")
    return norm


# ---------------------------------------------------------- reconciliación contable


def _signature(observed: float, expected: float) -> str | None:
    """¿La lectura es exactamente -1x (signo invertido) o 10x (decimal corrido) de lo esperado?"""
    if not expected:
        return None
    ratio = observed / expected
    for name, k in (("sign_flip", cfg.SIGN_FLIP_RATIO), ("decimal_shift_x10", cfg.DECIMAL_SHIFT_RATIO)):
        if abs(ratio - k) <= cfg.SIGNATURE_RTOL * abs(k):
            return name
    return None


def reconcile_account(
    cands: list[list[float]], inflow: np.ndarray, outflow: np.ndarray, tol: float = cfg.RECONCILIATION_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resuelve una cuenta día a día con la identidad contable.

    ``cands[t]`` lista los balances reportados ese día (0, 1 o 2 si hay duplicado).
    Reglas en orden: (1) verificar contra vecinos, (2) elegir el duplicado que cuadra,
    (3) rellenar nulos por la identidad, (4) recuperar un flujo faltante si ambos balances
    son confiables, (5) reparar solo firmas exactas (-1x, 10x), (6) aceptar con marca lo
    plausible pero no verificable, (7) cuarentena de lo demás.
    Devuelve ``(balance, balance_status, inflow, outflow, flow_status)``.
    """
    n = len(cands)
    bal = np.full(n, np.nan)
    status = np.array([""] * n, dtype=object)
    flow_status = np.array([""] * n, dtype=object)
    contradicts = np.zeros(n, dtype=bool)
    inflow, outflow = inflow.copy(), outflow.copy()

    def known(x: float) -> bool:
        return not np.isnan(x)

    def expected(t: int) -> tuple[float | None, str | None]:
        if t > 0 and known(bal[t - 1]) and known(inflow[t]) and known(outflow[t]):
            return bal[t - 1] + inflow[t] - outflow[t], "forward"
        if t < n - 1 and known(bal[t + 1]) and known(inflow[t + 1]) and known(outflow[t + 1]):
            return bal[t + 1] - inflow[t + 1] + outflow[t + 1], "backward"
        return None, None

    def verify_against_reported_neighbours() -> None:
        for t in range(n):
            if known(bal[t]):
                continue
            for c in cands[t]:
                ok = False
                if t > 0 and known(inflow[t]) and known(outflow[t]):
                    ok |= any(abs(c - (p + inflow[t] - outflow[t])) <= tol for p in cands[t - 1])
                if t < n - 1 and known(inflow[t + 1]) and known(outflow[t + 1]):
                    ok |= any(abs(nb - (c + inflow[t + 1] - outflow[t + 1])) <= tol for nb in cands[t + 1])
                if ok:
                    bal[t], status[t] = c, "observed_reconciled"
                    break

    def propagate() -> None:
        for _ in range(cfg.RECON_MAX_PASSES):
            changed = False
            verify_against_reported_neighbours()
            for t in range(n):
                if known(bal[t]):
                    continue
                exp, how = expected(t)
                if exp is None:
                    continue
                if not cands[t]:
                    bal[t], status[t], changed = exp, f"imputed_{how}", True
                elif any(abs(c - exp) <= tol for c in cands[t]):
                    bal[t] = min(cands[t], key=lambda c: abs(c - exp))
                    status[t], changed = "observed_reconciled", True
                else:
                    sig = _signature(cands[t][0], exp)
                    if sig:
                        bal[t], status[t], changed = exp, f"repaired_{sig}", True
                    else:
                        contradicts[t] = True
            for t in range(1, n):
                if known(bal[t]) and known(bal[t - 1]):
                    net = bal[t] - bal[t - 1]
                    if not known(inflow[t]) and known(outflow[t]) and net + outflow[t] >= -tol:
                        inflow[t], flow_status[t], changed = max(net + outflow[t], 0.0), "inflow_recovered", True
                    elif not known(outflow[t]) and known(inflow[t]) and inflow[t] - net >= -tol:
                        outflow[t], flow_status[t], changed = max(inflow[t] - net, 0.0), "outflow_recovered", True
            if not changed:
                break

    propagate()
    for _ in range(5):  # regla 6: aceptar con marca lo plausible pero no verificable
        accepted = False
        for i in range(n):
            if known(bal[i]) or not cands[i] or contradicts[i]:
                continue
            nearest = [j for j in sorted(range(n), key=lambda j: abs(j - i)) if known(bal[j])]
            if nearest and bal[nearest[0]] and abs(cands[i][0] / bal[nearest[0]] - 1) <= cfg.PLAUSIBLE_REL_CHANGE:
                bal[i], status[i], accepted = cands[i][0], "observed_unverified", True
        if not accepted:
            break
        propagate()
    for i in range(n):  # regla 7
        if not known(bal[i]):
            status[i] = "unreconciled_episode" if cands[i] else "unresolved_missing"
    return bal, status, inflow, outflow, flow_status


def reconcile_ledger(df: pd.DataFrame, tol: float = cfg.RECONCILIATION_TOL) -> pd.DataFrame:
    """Aplica :func:`reconcile_account` a cada cuenta.

    ``df``: columnas ``account_id, date_parsed, balance, inflow, outflow`` (fechas ya parseadas).
    Devuelve un registro por (cuenta, día de calendario), incluyendo los candidatos originales.
    """
    out = []
    for acc, g in df.groupby("account_id"):
        days = pd.date_range(g["date_parsed"].min(), g["date_parsed"].max(), freq="D")
        by_day = g.groupby("date_parsed")
        cands = by_day["balance"].apply(lambda s: s.dropna().tolist()).reindex(days)
        cands = [c if isinstance(c, list) else [] for c in cands]
        flows = by_day[["inflow", "outflow"]].first().reindex(days)
        bal, status, infl, outf, fstat = reconcile_account(
            cands, flows["inflow"].to_numpy(float), flows["outflow"].to_numpy(float), tol
        )
        out.append(
            pd.DataFrame(
                {
                    "account_id": acc,
                    "date": days,
                    "balance_clean": bal,
                    "balance_status": status,
                    "inflow_clean": infl,
                    "outflow_clean": outf,
                    "flow_status": fstat,
                    "inflow_raw": flows["inflow"].to_numpy(float),
                    "outflow_raw": flows["outflow"].to_numpy(float),
                    "n_rows": by_day.size().reindex(days).fillna(0).astype(int).to_numpy(),
                    "n_candidates": [len(c) for c in cands],
                    "cand_1": [c[0] if len(c) > 0 else np.nan for c in cands],
                    "cand_2": [c[1] if len(c) > 1 else np.nan for c in cands],
                }
            )
        )
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- transferencias


def clean_transfers(transfers: pd.DataFrame, accounts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida el log, deriva tipo/lag/fee y separa duplicados exactos (mismo todo salvo el id).

    Devuelve ``(limpio, excluidos)``; los excluidos se reportan, no se pierden.
    """
    ccy = accounts.set_index("account_id")["currency"]
    t = transfers.copy()
    if t["transfer_id"].duplicated().any():
        raise ValueError("transfer_id duplicados")
    t["from_ccy"] = t["from_account"].map(ccy)
    t["to_ccy"] = t["to_account"].map(ccy)
    if (t["currency"] != t["from_ccy"]).any():
        raise ValueError("`currency` del log no coincide con la moneda de la cuenta origen")
    if (t["from_account"] == t["to_account"]).any():
        raise ValueError("transferencia de una cuenta a sí misma")
    t["cross_ccy"] = t["from_ccy"] != t["to_ccy"]
    t["kind"] = np.where(t["cross_ccy"], "cross_ccy", "same_ccy")
    t["lag_days"] = (t["date_settled"] - t["date_requested"]).dt.days
    t["fee_rate"] = t["fee"] / t["amount"]
    if (t["lag_days"] < 0).any() or (t["fee"] < 0).any() or (t["amount"] <= 0).any():
        raise ValueError("lag/fee/monto fuera de rango")
    key = [c for c in transfers.columns if c != "transfer_id"]
    is_dup = t.sort_values("transfer_id").duplicated(subset=key, keep="first").reindex(t.index)
    return t[~is_dup].reset_index(drop=True), t[is_dup].reset_index(drop=True)


def transfer_memo_columns(panel: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Agrega, a la fecha de settlement, lo que sale (moneda propia) y lo que entra (a FX de config).

    Son columnas MEMO: los datos muestran que el log de transferencias no se refleja en
    ``inflow/outflow`` (ver features.reconciliation_report), así que no entran en ``net_flow``.
    """
    fx = cfg.FX_PER_USD
    out = transfers.groupby(["from_account", "date_settled"])["amount"].sum().rename("xfer_out_native")
    incoming = transfers.assign(
        received=transfers["amount"] * transfers["to_ccy"].map(fx) / transfers["from_ccy"].map(fx)
    )
    inn = incoming.groupby(["to_account", "date_settled"])["received"].sum().rename("xfer_in_native_at_cfg_fx")
    p = panel.merge(out, left_on=["account_id", "date"], right_index=True, how="left")
    p = p.merge(inn, left_on=["account_id", "date"], right_index=True, how="left")
    p[["xfer_out_native", "xfer_in_native_at_cfg_fx"]] = p[["xfer_out_native", "xfer_in_native_at_cfg_fx"]].fillna(0.0)
    return p


# ------------------------------------------------------------------------ panel

EXACT_BALANCE = {"observed_reconciled", "imputed_forward", "imputed_backward", "repaired_sign_flip", "repaired_decimal_shift_x10"}


def _typical_outflow(outflow: pd.Series, usable: pd.Series) -> pd.Series:
    """Mediana del egreso del mismo día de semana en ±ventana, usando solo filas confiables."""
    good = outflow.where(usable)
    dow = good.index.dayofweek
    result = {}
    span = pd.Timedelta(weeks=cfg.WEEKDAY_IMPUTE_WINDOW_WEEKS)
    for d in good.index:
        w = good[(dow == d.dayofweek) & (good.index >= d - span) & (good.index <= d + span)].dropna()
        result[d] = w.median() if len(w) else good.dropna().median()
    return pd.Series(result)


def build_panel(ledger: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Panel cuenta×día completo (0 nulos) e identidad contable exacta.

    * Cuarentena: los episodios sin explicación se sustituyen por un puente lineal entre el
      último saldo confiable y el primero posterior; los flujos de esa ventana se rehacen para
      cumplir la identidad. Todo queda marcado (``balance_status``/``flow_status``/``reliable``).
    * Flujos brutos irrecuperables (ambos nulos): egreso típico del día de semana y el ingreso
      se despeja de ``net_flow``. ``net_flow = diff(balance)`` siempre.
    """
    frames = []
    for acc, g in ledger.sort_values("date").groupby("account_id"):
        g = g.set_index("date")
        bal = g["balance_clean"].copy()
        status = g["balance_status"].copy()
        infl, outf = g["inflow_clean"].copy(), g["outflow_clean"].copy()
        fstat = g["flow_status"].replace("", "observed")

        override = pd.Series(False, index=g.index)
        quarantined = status == "unreconciled_episode"
        if quarantined.any():
            run_id = (quarantined != quarantined.shift()).cumsum()
            for _, run in g[quarantined].groupby(run_id[quarantined]):
                s, e = run.index.min(), run.index.max()
                before, after = s - pd.Timedelta(days=1), e + pd.Timedelta(days=1)
                if before not in g.index or after not in g.index or np.isnan(bal[before]) or np.isnan(bal[after]):
                    raise ValueError(f"{acc}: episodio {s.date()}–{e.date()} sin saldos confiables a ambos lados")
                span = pd.date_range(before, after)
                bal[span] = np.linspace(bal[before], bal[after], len(span))
                status[s:e] = "quarantine_bridged"
                override[s:after] = True
        if bal.isna().any():
            raise ValueError(f"{acc}: saldos sin resolver {list(bal.index[bal.isna()].date)}")

        net = bal.diff()
        typical = _typical_outflow(outf, usable=~override & outf.notna())
        # primer día: no hay saldo previo, el neto sale de los flujos
        first = bal.index[0]
        if override[first]:
            raise ValueError(f"{acc}: cuarentena en el primer día")
        o0 = outf[first] if not np.isnan(outf[first]) else typical[first]
        i0 = infl[first] if not np.isnan(infl[first]) else o0
        if np.isnan(outf[first]) or np.isnan(infl[first]):
            fstat[first] = "imputed_typical"
        net[first] = i0 - o0
        infl[first], outf[first] = i0, o0

        need = override | infl.isna() | outf.isna()
        need[first] = False
        for d in bal.index[need]:
            known_in = not override[d] and not np.isnan(infl[d])
            known_out = not override[d] and not np.isnan(outf[d])
            if known_in and not known_out:
                i_d, o_d = infl[d], infl[d] - net[d]
                fstat[d] = "derived_from_net"
            elif known_out and not known_in:
                o_d, i_d = outf[d], net[d] + outf[d]
                fstat[d] = "derived_from_net"
            else:  # ambos desconocidos: egreso típico del día de semana; el ingreso cierra la identidad
                o_d, i_d = typical[d], net[d] + typical[d]
                fstat[d] = "bridged_quarantine" if override[d] else "imputed_typical"
            if i_d < 0:
                i_d, o_d = 0.0, -net[d]
            elif o_d < 0:
                o_d, i_d = 0.0, net[d]
            infl[d], outf[d] = i_d, o_d

        frame = pd.DataFrame(
            {
                "account_id": acc,
                "date": bal.index,
                "balance": bal.to_numpy(),
                "inflow": infl.to_numpy(),
                "outflow": outf.to_numpy(),
                "net_flow": net.to_numpy(),
                "balance_status": status.to_numpy(),
                "flow_status": fstat.to_numpy(),
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["reliable"] = panel["balance_status"].isin(EXACT_BALANCE) & panel["flow_status"].isin(
        {"observed", "inflow_recovered", "outflow_recovered"}
    )
    meta = accounts[["account_id", "bank_name", "currency", "account_type"]]
    return meta.merge(panel, on="account_id").sort_values(["account_id", "date"]).reset_index(drop=True)


def validate_panel(panel: pd.DataFrame) -> None:
    """Criterios de aceptación de F1: falla si el panel no cumple el contrato."""
    if len(panel) != cfg.EXPECTED_PANEL_ROWS:
        raise AssertionError(f"filas {len(panel)} != {cfg.EXPECTED_PANEL_ROWS}")
    if panel.isna().any().any():
        raise AssertionError(f"nulos en el panel: {panel.isna().sum()[panel.isna().sum() > 0].to_dict()}")
    if (panel["balance"] < 0).any():
        raise AssertionError("balances negativos en el panel")
    for acc, g in panel.groupby("account_id"):
        if len(g) != cfg.EXPECTED_DAYS or g["date"].diff().dropna().ne(pd.Timedelta(days=1)).any():
            raise AssertionError(f"{acc}: huecos de calendario")
    resid = panel.groupby("account_id")["balance"].diff() - panel["net_flow"]
    resid = resid.dropna().abs()
    if resid.max() > cfg.RECONCILIATION_TOL:
        raise AssertionError(f"identidad contable rota: residuo máximo {resid.max()}")
    ident = (panel["inflow"] - panel["outflow"] - panel["net_flow"]).abs()
    if ident.max() > cfg.RECONCILIATION_TOL:
        raise AssertionError("inflow - outflow != net_flow")


# --------------------------------------------------------------------- pipeline


def run_pipeline(write: bool = True) -> dict[str, pd.DataFrame]:
    """Ejecuta F1 completo y devuelve panel, ledger, transferencias y evidencias."""
    accounts = load_accounts()
    raw = load_raw_balances()
    if len(raw) != cfg.EXPECTED_RAW_ROWS:
        raise AssertionError(f"filas crudas {len(raw)} != {cfg.EXPECTED_RAW_ROWS}")
    raw["date_parsed"] = parse_mixed_dates(raw["date"])
    raw["currency_norm"] = normalize_currency(raw["currency"], raw["account_id"], accounts)
    ledger = reconcile_ledger(raw)
    panel = build_panel(ledger, accounts)

    transfers_raw = load_raw_transfers()
    transfers, excluded = clean_transfers(transfers_raw, accounts)
    panel = transfer_memo_columns(panel, transfers)
    validate_panel(panel)

    if write:
        cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        panel.to_csv(cfg.PANEL_FILE, index=False)
        transfers.to_csv(cfg.TRANSFERS_CLEAN_FILE, index=False)
    return {
        "accounts": accounts, "raw": raw, "ledger": ledger, "panel": panel,
        "transfers": transfers, "transfers_excluded": excluded, "transfers_raw": transfers_raw,
    }


if __name__ == "__main__":
    from .quality import quality_report

    res = run_pipeline()
    print(quality_report(res["raw"], res["ledger"], res["panel"], res["transfers_raw"], res["transfers_excluded"]).to_string())
    print(f"\npanel: {res['panel'].shape} -> {cfg.PANEL_FILE}")
