"""Política D (cuantil + espera): se comporta como B cuando no espera, ahorra transferencias cuando espera y explica la espera."""
import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src import forecast as F
from src import ingest, policy as P


@pytest.fixture(scope="module")
def setup():
    panel = ingest.run_pipeline(write=False)["panel"]
    book = P.Book.from_panel(panel)
    store = B.ForecastStore(F.net_flow_matrix(panel), "empirical")
    return book, store


def _plan(res):
    return [(s.request_idx, s.i, s.j, round(s.amount, 2)) for s, _ in res.transfers]


def test_quantile_policy_refactor_keeps_its_results(setup):
    book, store = setup
    t = B.totals_usd(B.simulate(book, P.QuantilePolicy(), store))
    assert t["n_transferencias"] == 49 and t["dias_cuenta_en_faltante"] == 0 and round(t["costo_total_usd"]) == 4955


def test_wait_policy_equals_B_when_waiting_is_disabled(setup):
    book, store = setup
    base = _plan(B.simulate(book, P.QuantilePolicy(), store))
    no_slack = P.WaitAwarePolicy(min_slack_days=10_000)                  # nunca hay holgura suficiente
    no_recovery = P.WaitAwarePolicy(recovery_margin_days=1e6)            # la mediana nunca "se recupera"
    assert _plan(B.simulate(book, no_slack, store)) == base and not no_slack.log
    assert _plan(B.simulate(book, no_recovery, store)) == base and not no_recovery.log


def test_waiting_reduces_transfers_and_explains_itself(setup):
    book, store = setup
    pol = P.WaitAwarePolicy()
    res = B.simulate(book, pol, store)
    b = B.totals_usd(B.simulate(book, P.QuantilePolicy(), store))
    d = B.totals_usd(res)
    assert d["n_transferencias"] < b["n_transferencias"] and d["costo_total_usd"] < b["costo_total_usd"]
    assert pol.log and all(x["holgura_dias"] >= pol.min_slack_days for x in pol.log)
    e = pol.log[0]
    assert pd.Timestamp(e["ultimo_dia_para_solicitar"]) > pd.Timestamp(e["fecha"])        # siempre queda al menos un día para actuar


def test_opportunities_table_is_consistent(setup):
    book, store = setup
    t = list(book.dates).index(pd.Timestamp("2025-09-26"))
    pol = P.WaitAwarePolicy()
    tab = pol.opportunities(B.context_at(book, store, t))
    assert len(tab) == 6 and set(tab["estado"]) <= {"ok", "esperar", "transferir"}
    w = tab[tab["estado"] == "esperar"]
    if len(w):                                                                             # espera: el último día para solicitar cae antes del faltante
        assert (pd.to_datetime(w["ultimo_dia_para_solicitar"]) <= pd.to_datetime(w["primer_dia_bajo_piso"])).all()
    assert tab.loc[tab["estado"] == "ok", "primer_dia_bajo_piso"].isna().all()


def test_book_extended_gives_tomorrows_floor_without_inventing_data(setup):
    book, _ = setup
    bx = book.extended(1)
    T = len(book.dates)
    assert bx.balance.shape == (T + 1, book.n) and bx.dates[-1] == book.dates[-1] + pd.Timedelta(days=1)
    assert np.allclose(bx.balance[:T], book.balance) and np.isnan(bx.balance[T]).all()          # no hay datos futuros
    expected_avg = book.outflow[-28:].mean(axis=0)                                              # egreso medio de los 28 días que terminan hoy
    assert np.allclose(bx.avg_outflow[T], expected_avg)
    k = book.floor[-1] / book.avg_outflow[-1]
    assert np.allclose(bx.floor[T], expected_avg * k)                                           # mismo K de cada cuenta (operativa/reserva)
    assert bx.date_at(T) == bx.dates[-1]
