"""F4/F5: políticas, MILP y simulador (conservación, FX, lag, costo fijo)."""
import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src import config as cfg
from src import forecast as F
from src import optimize as O
from src import policy as P


@pytest.fixture(scope="module")
def world():
    panel = pd.read_csv(cfg.PANEL_FILE, parse_dates=["date"])
    book = P.Book.from_panel(panel)
    store = B.ForecastStore(F.net_flow_matrix(panel), "empirical")
    return book, store


def _ctx(book, store, t, bal=None):
    return P.Context(t, book, book.balance[t].copy() if bal is None else bal, book.balance[: t + 1], [], store[t], np.random.default_rng(1))


def test_book_floor_has_no_lookahead(world):
    book, _ = world
    t = 100
    manual = book.outflow[t - 28 : t].mean(axis=0) * 25.0
    assert np.allclose(book.floor[t], manual)


def test_fx_ratio_and_kinds(world):
    book, _ = world
    i, j = book.accounts.index("ACC-004"), book.accounts.index("ACC-005")   # MXN → MXN
    assert book.kind(i, j) == "same_ccy" and book.fx_ratio(i, j) == 1.0 and book.lag_plan(i, j) == cfg.LAG_PLAN["same_ccy"]
    k = book.accounts.index("ACC-002")                                    # MXN → USD
    assert book.kind(i, k) == "cross_ccy"
    assert np.isclose(book.fx_ratio(i, k), 1 / cfg.FX_PER_USD["MXN"])


def test_squad_rule_reproduces_the_flaw(world):
    """El squad elige donante por saldo nominal: siempre una cuenta COP, con el monto sin convertir."""
    book, store = world
    props = P.SquadRule().decide(_ctx(book, store, 150))
    assert props and all(book.ccy[p.i] == "COP" for p in props)


def test_quantile_policy_plan_respects_fx_lag_and_donor_floor(world):
    book, store = world
    t = 150
    ctx = _ctx(book, store, t)
    bal = ctx.bal.copy()
    a = book.accounts.index("ACC-004")
    bal[a] = book.floor[t + 1, a] * 0.6                                   # forzamos un faltante en la operativa MXN
    ctx = _ctx(book, store, t, bal)
    pol = P.QuantilePolicy()
    props = pol.decide(ctx)
    assert any(p.j == a for p in props), "debería proponer una transferencia hacia la cuenta en faltante"
    plan = P.plan_to_frame(props, ctx)
    for p, (_, row) in zip(props, plan.iterrows()):
        assert np.isclose(row["monto_recibido"], p.amount * book.fx_ratio(p.i, p.j))                # FX aplicado
        assert pd.Timestamp(row["liquidacion_esperada"]) >= pd.Timestamp(row["fecha_ejecucion"]) + pd.Timedelta(days=book.lag_plan(p.i, p.j))
        assert p.amount * (1 + book.fee_rate(p.i, p.j)) <= bal[p.i] - book.floor[t + 1, p.i]      # el donante no cae bajo su piso
    assert plan["fecha_ejecucion"].iloc[0] == (book.dates[t] + pd.Timedelta(days=1)).date()


def test_milp_transfers_when_deficit_is_large_and_skips_when_operating_costs_more(world):
    book, store = world
    t = 150
    a = book.accounts.index("ACC-004")
    ctx = _ctx(book, store, t)
    big = ctx.bal.copy(); big[a] = book.floor[t + 1, a] * 0.5
    pol = O.LPPolicy()
    assert pol.decide(_ctx(book, store, t, big)), "déficit grande → transferir"
    small = 3.0 * book.floor[t + 1]                                         # todas holgadas...
    small[a] = book.floor[t + 1, a] * 1.0005                                # ...salvo ACC-004, en el piso
    pol2 = O.LPPolicy(fixed_cost=1e6)                                       # operar cuesta más que cualquier penalización evitable
    out = pol2.decide(_ctx(book, store, t, small))
    assert out == [] and len(pol2.log) >= 1
    assert all(r["penalizacion_esperada_usd"] < r["costo_minimo_de_cubrir_usd"] for r in pol2.log if r["cuenta"] == "ACC-004")


def test_simulator_conserves_money_and_applies_fees(world):
    """Σ (saldo simulado − orgánico) en USD = − fees − monto aún en tránsito (a FX de config)."""
    book, store = world
    res = B.simulate(book, P.QuantilePolicy(), store)
    usd = 1 / book.fx
    diff = ((res.balance[-1] - book.balance[-1]) * usd).sum()
    fees = sum(s.fee * usd[s.i] for s, _ in res.transfers)
    in_transit = sum(s.amount * usd[s.i] for s, _ in res.transfers if s.settle_idx >= len(book.dates))
    assert np.isclose(diff, -fees - in_transit, rtol=1e-6, atol=1e-6)


def test_transfer_never_lands_before_lag_and_cash_is_available(world):
    book, store = world
    res = B.simulate(book, P.QuantilePolicy(), store)
    for s, _ in res.transfers:
        kind = book.kind(s.i, s.j)
        assert s.settle_idx - s.request_idx in cfg.LAG_OBSERVED[kind]
    assert res.balance.min() >= 0


def test_forced_lag_and_fx_shock_change_credits(world):
    book, store = world
    base = B.simulate(book, P.QuantilePolicy(), store)
    stress = B.simulate(book, P.QuantilePolicy(), store, B.SimSettings(lag_forced=3))
    assert all(s.settle_idx - s.request_idx == 3 for s, _ in stress.transfers)
    shock = B.simulate(book, P.QuantilePolicy(), store, B.SimSettings(fx_shock=0.1))
    cross = [(a, b) for (a, _), (b, _) in zip(base.transfers, shock.transfers) if book.kind(a.i, a.j) == "cross_ccy" and a.i == b.i and a.j == b.j and np.isclose(a.amount, b.amount)]
    assert cross and any(not np.isclose(a.credit, b.credit) for a, b in cross)


def test_fixed_cost_counted_per_operation(world):
    book, store = world
    res = B.simulate(book, P.QuantilePolicy(), store)
    tot = B.totals_usd(res)
    assert np.isclose(tot["costo_fijo_usd"], len(res.transfers) * cfg.FIXED_COST_USD)
