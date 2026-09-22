"""F3: baseline, ausencia de lookahead, pinball y regla de selección."""
import numpy as np
import pandas as pd
import pytest

from src import forecast as F


@pytest.fixture(scope="module")
def flows():
    idx = pd.date_range("2025-01-01", periods=120)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"A": rng.normal(0, 1, 120), "B": rng.normal(-0.2, 2, 120)}, index=idx)


def test_seasonal_naive_repeats_last_week(flows):
    f = F.fit_origin(flows, 99, "seasonal_naive", H=14)
    last = flows["A"].iloc[93:100].to_numpy()
    assert np.allclose(f.mean[:7, 0], last) and np.allclose(f.mean[7:14, 0], last)


def test_prophet_forecasts_h_steps_without_nan(flows):
    f = F.fit_origin(flows, 99, "prophet", H=14)
    assert f.mean.shape == (14, 2)
    assert np.isfinite(f.mean).all()


def test_no_lookahead(flows):
    """Cambiar datos posteriores al origen no puede alterar el forecast."""
    other = flows.copy()
    other.iloc[100:] = 1e6
    for m in ("empirical", "seasonal_naive", "ets", "sarimax", "prophet"):
        a, b = F.fit_origin(flows, 99, m), F.fit_origin(other, 99, m)
        assert np.allclose(a.mean, b.mean), m
        assert all(np.allclose(x[0], y[0]) and np.allclose(x[1], y[1]) for x, y in zip(a.pools, b.pools)), m


def test_draws_have_expected_shape_and_are_reproducible(flows):
    f = F.fit_origin(flows, 99, "empirical")
    a = f.draw(50, np.random.default_rng(1))
    b = f.draw(50, np.random.default_rng(1))
    assert a.shape == (50, 14, 2) and np.allclose(a, b)


def test_pinball_known_values():
    y = np.array([1.0])
    q = np.array([[0.0, 2.0]])            # cuantiles 0.1 y 0.9
    out = F.pinball(y, q, taus=(0.1, 0.9))
    assert np.allclose(out, [[0.1 * 1.0, 0.1 * 1.0]])     # 0.1*(1-0)  y  (0.9-1)*(1-2)=0.1


def test_walk_forward_scores_have_coverage_between_0_and_1(flows):
    r = F.walk_forward(flows, "empirical", step=10, n_draws=200, min_train=60)
    s = F.score(r)
    assert set(s["h"]) == {7, 14} and s["cover_cum"].between(0, 1).all()


def test_selection_prefers_simplest_within_tolerance():
    rows = []
    for m, pin in (("empirical", 0.70), ("seasonal_naive", 1.0), ("ets", 0.69), ("sarimax", 0.60), ("prophet", 0.72)):
        for acc in ("A",):
            for h in (7, 14):
                rows.append({"model": m, "account_id": acc, "h": h, "mae_daily": pin, "pinball_daily": pin,
                             "mae_cum": pin, "pinball_cum": pin, "cover_cum": 0.9, "cover_daily": 0.9})
    best, _ = F.select_model(pd.DataFrame(rows))
    assert best == "sarimax"                           # 0.60 está a >2% de los demás: gana por métrica
    rows = [dict(r, pinball_cum=(0.61 if r["model"] == "empirical" else r["pinball_cum"])) for r in rows]
    best, _ = F.select_model(pd.DataFrame(rows))
    assert best == "empirical"                         # dentro de tolerancia → el más simple


def test_old_vs_new_forecast_reconciles_with_squad_and_d4():
    from src import backtest as B, diagnosis as D, forecast as F, ingest

    res = ingest.run_pipeline(write=False)
    panel = res["panel"]
    squad = D.squad_pipeline(res["accounts"], ingest.load_raw_balances())
    store = B.ForecastStore(F.net_flow_matrix(panel), "empirical")
    today, bt = D.old_vs_new_forecast(squad["balances"], panel, store)

    for acc, v in squad["forecasts"].items():                      # tu forecast_next_week sobre datos sin limpiar
        assert np.isclose(today.loc[acc, "squad_sin_limpiar"], v)
    ref, _ = D.level_forecast_backtest(panel, store)
    m = bt.merge(ref[["account_id", "h", "mae_squad"]], on=["account_id", "h"])
    assert np.allclose(m["mae_limpio"], m["mae_squad"])            # mismo criterio que D4
    assert (bt["mae_propuesto/sin_limpiar"] < 1).all()             # el método nuevo sobre datos limpios gana en las 12 combinaciones
