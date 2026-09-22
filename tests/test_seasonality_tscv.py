"""D4b — identificación de estacionalidad semanal y validación con TimeSeriesSplit."""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import diagnosis as D
from src import ingest, seasonality, tscv


@pytest.fixture(scope="module")
def result():
    return ingest.run_pipeline(write=False)


@pytest.fixture(scope="module")
def cv(result):
    squad = D.squad_pipeline(result["accounts"], ingest.load_raw_balances())
    return tscv.cv_level_forecasts(result["panel"], squad["balances"]), squad


# ---- identificación
def test_identify_period_finds_weekly_pattern_and_rejects_noise():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=270)
    weekly = pd.Series(rng.normal(size=270) + np.tile([2, 1, 1, 1, 1, -3, -3], 40)[:270], index=idx)
    assert seasonality.identify_period(weekly) == 7
    assert seasonality.identify_period(pd.Series(rng.normal(size=270), index=idx)) is None


def test_weekly_pattern_is_in_gross_flows_not_in_net_flow(result):
    rep = seasonality.weekly_seasonality_report(result["panel"])
    assert not rep.xs("net_flow", level="serie")["semanal"].any()          # el saldo (neto acumulado) no tiene patrón semanal
    for acc in ("ACC-001", "ACC-002", "ACC-003", "ACC-005", "ACC-006"):     # entradas y salidas sí
        assert rep.loc[(acc, "inflow"), "semanal"] and rep.loc[(acc, "outflow"), "semanal"]
    assert not rep.loc[("ACC-004", "outflow"), "semanal"]                   # excepción documentada


# ---- validación cruzada
def test_time_series_split_has_no_leakage(cv, result):
    (detail, _), _ = cv
    assert detail.groupby("fold")["date"].min().gt(detail.groupby("fold")["origen"].first()).all()   # prueba siempre después del origen
    blocks = detail.groupby("fold")["date"].agg(["min", "max"]).sort_index()
    assert (blocks["min"].iloc[1:].to_numpy() > blocks["max"].iloc[:-1].to_numpy()).all()            # bloques de prueba sin solaparse
    assert len(detail) == cfg.TSCV_N_SPLITS * cfg.TSCV_TEST_SIZE * 6 * len(tscv.MODELS)
    assert detail["date"].max() == result["panel"]["date"].max()                                     # el último bloque llega al final


def test_squad_baselines_match_their_definitions(cv, result):
    (detail, _), squad = cv
    panel = result["panel"]
    row = detail[(detail["fold"] == 3) & (detail["account_id"] == "ACC-004")]
    origin = row["origen"].iloc[0]
    limpio = panel[(panel["account_id"] == "ACC-004") & (panel["date"] <= origin)]["balance"].tail(cfg.SQUAD_WINDOW).mean()
    crudo = D.squad_raw_mean(squad["balances"][squad["balances"]["account_id"] == "ACC-004"].sort_values("date"), origin)
    assert np.isclose(row[row["model"] == "squad_limpio"]["forecast"].iloc[0], limpio)
    assert np.isclose(row[row["model"] == "squad_sin_limpiar"]["forecast"].iloc[0], crudo)


def test_gap_shifts_the_first_forecast_step(result):
    squad = D.squad_pipeline(result["accounts"], ingest.load_raw_balances())
    detail, _ = tscv.cv_level_forecasts(result["panel"], squad["balances"], n_splits=3, test_size=7, gap=3)
    assert detail["h"].min() == 4 and detail["h"].max() == 10          # gap = 3 -> el primer día de prueba es h = 4


def test_cv_summary_is_relative_to_the_squad_on_raw_data(cv):
    (detail, _), _ = cv
    s = tscv.cv_summary(detail)
    assert np.isclose(s["por_cuenta"].xs(tscv.BASE, level="model")["razon_vs_base"], 1).all()
    assert s["global"].loc["ultimo_saldo", "cuentas_que_mejoran_a_base"] == 6


# ---- ACF / PACF y decisión de diferenciar
def test_differencing_decision_on_synthetic_series():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=270)
    random_walk = pd.Series(np.cumsum(rng.normal(size=270)) - 0.05 * np.arange(270), index=idx)
    weekly = pd.Series(rng.normal(size=270) + np.tile([2, 1, 1, 1, 1, -3, -3], 40)[:270], index=idx)
    white = pd.Series(rng.normal(size=270), index=idx)
    rw, wk, wn = (seasonality.differencing_decision(x) for x in (random_walk, weekly, white))
    assert rw["d"] == 1 and not rw["sobrediferenciada"]              # paseo aleatorio: se diferencia y queda ruido blanco
    assert wk["d"] == 0 and wk["adf_p"] < cfg.ADF_ALPHA              # estacionaria con semana: ADF ya rechaza raíz unitaria
    assert wn["d"] == 0 and wn["sobrediferenciada"] and wn["acf1_dif"] < -0.4   # ruido blanco diferenciado: ACF(1) ~ -0,5


def test_acf_pacf_report_matches_the_findings(result):
    diff_t, seas_t, stationary = seasonality.acf_pacf_report(result["panel"])
    assert (diff_t.xs("balance", level="serie")["d"] == 1).all()             # el saldo es un paseo aleatorio con deriva
    assert (diff_t.drop("balance", level="serie")["d"] == 0).all()           # diferenciar los flujos los sobrediferencia
    weekly = {k for k, v in seas_t["semanal"].items() if v}
    assert weekly == {("ACC-001", "outflow"), ("ACC-003", "inflow"), ("ACC-003", "outflow"), ("ACC-005", "inflow"),
                      ("ACC-005", "outflow"), ("ACC-006", "inflow"), ("ACC-006", "outflow")}
    assert (seas_t["D"] == 0).all()                                           # sin diferenciación estacional: el patrón es determinístico
    assert seas_t["acf_7_tras_dif_7"].between(-0.6, -0.3).all()               # diferenciar en 7 sobrediferencia (~ -0,5)
    assert len(stationary) == 18


def test_fig_acf_pacf_builds_the_grid(result):
    import matplotlib

    matplotlib.use("Agg")
    from src import plots

    diff_t, _, stationary = seasonality.acf_pacf_report(result["panel"])
    for which in ("acf", "pacf"):
        fig = plots.fig_acf_pacf(stationary, diff_t, which)
        assert len(fig.axes) == 18


# ---- perfiles por ciclo (ver la estacionalidad antes de probarla)
def test_cycle_position_definitions():
    idx = pd.DatetimeIndex(["2025-03-03", "2025-03-15", "2025-03-16", "2025-03-31"])        # lunes, sábado, domingo, lunes
    assert list(seasonality.cycle_position(idx, "semanal")) == [0, 5, 6, 0]
    assert list(seasonality.cycle_position(idx, "quincenal")) == [3, 15, 1, 16]              # día 16 abre la 2ª quincena; el 31 es la posición 16
    assert list(seasonality.cycle_position(idx, "mensual")) == [3, 15, 16, 31]
    with pytest.raises(ValueError):
        seasonality.cycle_position(idx, "anual")


def test_cycle_profile_detects_weekly_pattern_and_not_noise():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2025-01-01", periods=270)
    weekly = pd.Series(100 + rng.normal(size=270) * 5 + np.where(idx.dayofweek >= 5, -25, 10), index=idx)   # cae en sábado y domingo
    noise = pd.Series(100 + rng.normal(size=270) * 5, index=idx)
    tab, p = seasonality.cycle_profile(weekly, "semanal")
    assert p < 1e-6 and len(tab) == 7 and tab["media"].iloc[5] < tab["media"].iloc[0]         # el fin de semana cae
    assert seasonality.cycle_profile(noise, "semanal")[1] > 0.01
    assert len(seasonality.cycle_profile(noise, "quincenal")[0]) == 16 and len(seasonality.cycle_profile(noise, "mensual")[0]) == 31


def test_cycle_pvalues_and_figure_on_real_data(result):
    import matplotlib

    matplotlib.use("Agg")
    from src import plots

    pv = seasonality.cycle_pvalues(result["panel"])
    assert pv.shape == (18, 3) and list(pv.columns) == list(seasonality.CYCLES)
    assert (pv.xs("balance", level="serie")["semanal"] > 0.05).all()                          # el saldo no tiene patrón semanal
    assert pv.loc[("ACC-003", "outflow"), "semanal"] < 1e-6
    assert (pv[["quincenal", "mensual"]] < 0.01).sum().sum() == 1                            # un único caso aislado (ACC-004, saldo, mensual)
    for serie in ("balance", "inflow", "outflow"):
        assert len(plots.fig_seasonality(result["panel"], serie).axes) == 18
