"""Un test por cada corrección de F1 (fecha, moneda, duplicado, sign flip, nulo) + contrato del panel."""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import ingest


@pytest.fixture(scope="module")
def result():
    return ingest.run_pipeline(write=False)


# ---- fechas
def test_parse_mixed_dates_recovers_all_formats():
    s = pd.Series(["2025-03-04", "24/06/2025", "04-13-2025"])
    out = ingest.parse_mixed_dates(s)
    assert list(out) == [pd.Timestamp("2025-03-04"), pd.Timestamp("2025-06-24"), pd.Timestamp("2025-04-13")]


def test_parse_mixed_dates_fails_loudly():
    with pytest.raises(ValueError):
        ingest.parse_mixed_dates(pd.Series(["2025/03/04"]))
    with pytest.raises(ValueError):
        ingest.parse_mixed_dates(pd.Series([None]))


def test_no_date_is_lost_on_real_data(result):
    assert result["raw"]["date_parsed"].notna().all()
    assert len(result["raw"]) == cfg.EXPECTED_RAW_ROWS


# ---- moneda
def test_currency_normalization_and_cross_validation():
    accounts = pd.DataFrame({"account_id": ["A", "B"], "currency": ["COP", "USD"]})
    out = ingest.normalize_currency(pd.Series(["Colombian Peso", "us dollar"]), pd.Series(["A", "B"]), accounts)
    assert list(out) == ["COP", "USD"]
    with pytest.raises(ValueError):
        ingest.normalize_currency(pd.Series(["USD"]), pd.Series(["A"]), accounts)  # cuenta COP con etiqueta USD
    with pytest.raises(ValueError):
        ingest.normalize_currency(pd.Series(["euro"]), pd.Series(["A"]), accounts)


# ---- duplicados: gana el que cuadra con la identidad, no el promedio
def test_duplicate_resolution_picks_row_that_reconciles():
    cands = [[100.0], [110.0, 111.5], [120.0]]      # día 1 duplicado; solo 110 cuadra con inflow/outflow
    inflow = np.array([0.0, 10.0, 10.0])
    outflow = np.array([0.0, 0.0, 0.0])
    bal, status, *_ = ingest.reconcile_account(cands, inflow, outflow)
    assert bal[1] == 110.0 and status[1] == "observed_reconciled"


# ---- sign flip: se repara por la identidad, no con abs()
def test_sign_flip_repaired_by_identity_not_abs():
    cands = [[100.0], [105.0], [-110.0], [115.0], [120.0]]   # el día 2 real es 110
    inflow = np.array([0.0, 5.0, 5.0, 5.0, 5.0])
    outflow = np.zeros(5)
    bal, status, *_ = ingest.reconcile_account(cands, inflow, outflow)
    assert bal[2] == 110.0 and status[2] == "repaired_sign_flip"


def test_negative_that_is_not_a_sign_flip_is_not_repaired():
    cands = [[100.0], [105.0], [-40.0], [115.0], [120.0]]    # -40 no es -1x de 110
    inflow = np.array([0.0, 5.0, 5.0, 5.0, 5.0])
    outflow = np.zeros(5)
    bal, status, *_ = ingest.reconcile_account(cands, inflow, outflow)
    assert status[2] != "repaired_sign_flip" and not (bal[2] > 0 and status[2].startswith("repaired"))


# ---- nulos: balance por identidad, flujo por diferencia de balances
def test_null_balance_and_null_flow_recovered_exactly():
    cands = [[90.0], [100.0], [], [130.0], [135.0]]
    inflow = np.array([0.0, 10.0, 20.0, np.nan, 5.0])
    outflow = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    bal, status, infl, outf, fstat = ingest.reconcile_account(cands, inflow, outflow)
    assert bal[2] == 115.0 and status[2].startswith("imputed")
    assert infl[3] == 20.0 and fstat[3] == "inflow_recovered"      # 130 = 115 + in - 5


# ---- contrato del panel (criterios de aceptación F1)
def test_panel_contract(result):
    panel = result["panel"]
    ingest.validate_panel(panel)
    assert len(panel) == 1620
    assert (panel["balance"] >= 0).all()
    assert not panel.isna().any().any()


def test_panel_matches_negative_balance_ground_truth(result):
    panel = result["panel"]
    for acc, day in [("ACC-005", "2025-04-10"), ("ACC-001", "2025-07-18"), ("ACC-003", "2025-06-24"), ("ACC-003", "2025-04-03")]:
        row = panel[(panel["account_id"] == acc) & (panel["date"] == day)].iloc[0]
        assert row["balance"] > 0 and row["balance_status"] == "repaired_sign_flip"


def test_identity_holds_everywhere(result):
    p = result["panel"].sort_values(["account_id", "date"])
    r = (p["balance"] - (p.groupby("account_id")["balance"].shift(1) + p["inflow"] - p["outflow"])).dropna().abs()
    assert r.max() <= cfg.RECONCILIATION_TOL


def test_quarantine_is_flagged_not_hidden(result):
    p = result["panel"]
    q = p[p["balance_status"] == "quarantine_bridged"]
    assert len(q) == 18 and not q["reliable"].any()


# ---- transferencias
def test_transfer_regimes_and_duplicates(result):
    t = result["transfers"]
    assert len(result["transfers_excluded"]) == 5 and len(t) == 140
    same = t[t["kind"] == "same_ccy"]
    assert np.allclose(same["fee_rate"], cfg.FEE_RATE["same_ccy"], atol=2e-5)
    assert set(same["lag_days"]) <= {0, 1}
    assert set(t[t["kind"] == "cross_ccy"]["lag_days"]) <= {1, 2, 3}


def test_cleaning_diff_matches_quality_report(result):
    from src import quality

    t = quality.cleaning_diff(result["raw"], result["ledger"], result["panel"])
    n = t["problema"].value_counts()
    assert n["signo invertido"] == 4 and n["decimal corrido x10"] == 3
    assert n["cuarentena (puente lineal)"] == 18 and n["duplicado con balance distinto"] == 44
    corrupt = t[t["problema"].isin(["signo invertido", "decimal corrido x10", "cuarentena (puente lineal)"])]
    assert (corrupt["desvio_pct"] > 1).all()          # toda corrupción se aparta > 1% del valor limpio
    assert (t["balance_limpio"] > 0).all()


def test_cleaning_summary_before_after(result):
    from src import diagnosis as D, quality

    t = quality.cleaning_summary(result["raw"], result["panel"])
    m = lambda prefix: t.xs(next(k for k in t.index.get_level_values(1).unique() if k.startswith(prefix)), level=1)
    assert (m("filas")["después"] == cfg.EXPECTED_DAYS).all() and (m("balances nulos")["después"] == 0).all()
    assert int(m("lecturas corruptas")["antes"].sum()) == 23           # las que el squad ve (25 en total; 2 caen en fechas que pierde)
    squad = D.squad_pipeline(result["accounts"], ingest.load_raw_balances())
    for acc, v in m("forecast del squad")["antes"].items():
        assert np.isclose(v, squad["forecasts"][acc])                   # mismo forecast que el pipeline del squad
