"""D4d — XGBoost con rezagos: sin fuga de datos, mismos cortes que el resto de modelos y métricas bien definidas."""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import diagnosis as D
from src import ingest, ml_forecast as ML, tscv


@pytest.fixture(scope="module")
def result():
    return ingest.run_pipeline(write=False)


@pytest.fixture(scope="module")
def detail_x(result):
    return ML.cv_xgb(result["panel"])


def test_error_table_definitions(result):
    panel = result["panel"]
    origin = panel["date"].sort_values().iloc[100]
    real = 1_000.0
    detail = pd.DataFrame({
        "fold": 1, "origen": origin, "account_id": "ACC-002", "model": "m", "h": [1, 2, 3], "date": origin,
        "forecast": [real * 1.20, real * 0.80, real * 1.05], "real": real,
    }).assign(error=lambda d: d["forecast"] - d["real"])
    t = ML.error_table(detail, panel).loc["m"]
    assert np.isclose(t["MAPE %"], (20 + 20 + 5) / 3)
    assert np.isclose(t["sMAPE %"], np.mean([200 * 0.2 / 2.2, 200 * 0.2 / 1.8, 200 * 0.05 / 2.05]))
    assert np.isclose(t["sobreestima % (>+15%)"], 100 / 3) and np.isclose(t["subestima % (<-15%)"], 100 / 3)
    assert np.isclose(t["balanceado % (±15%)"], 100 / 3)
    assert np.isclose(t["sesgo % (+ = sobreestima)"], (20 - 20 + 5) / 3)


def test_xgb_uses_the_same_folds_as_the_other_models(result, detail_x):
    squad = D.squad_pipeline(result["accounts"], ingest.load_raw_balances())
    base, _ = tscv.cv_level_forecasts(result["panel"], squad["balances"])
    keys = ["fold", "origen", "account_id", "h", "date"]
    a = base[base["model"] == "ultimo_saldo"][keys].reset_index(drop=True)
    b = detail_x[keys].reset_index(drop=True)
    assert a.equals(b)
    assert len(detail_x) == cfg.TSCV_N_SPLITS * cfg.TSCV_TEST_SIZE * 6


def test_xgb_forecast_has_no_lookahead(result, detail_x):
    """Alterar TODO lo que ocurre después del origen del corte 1 no puede cambiar sus pronósticos."""
    panel = result["panel"].copy()
    origin = detail_x[detail_x["fold"] == 1]["origen"].iloc[0]
    future = panel["date"] > origin
    rng = np.random.default_rng(0)
    for col in ("balance", "inflow", "outflow", "net_flow"):
        panel.loc[future, col] = panel.loc[future, col] * rng.uniform(0.2, 5.0, future.sum())
    altered = ML.cv_xgb(panel)
    f1 = lambda d: d[d["fold"] == 1].sort_values(["account_id", "h"])["forecast"].to_numpy()
    assert np.allclose(f1(detail_x), f1(altered))                       # corte 1: idéntico
    assert not np.allclose(detail_x[detail_x["fold"] == 5]["forecast"].to_numpy(),
                           altered[altered["fold"] == 5]["forecast"].to_numpy())   # el control: en otros cortes el futuro sí cuenta


def test_feature_importance_sums_to_at_most_one(result):
    imp = ML.feature_importance(result["panel"])
    assert 0 < imp.sum() <= 1 + 1e-9 and len(imp) == 12


# --------------------------------------------------------------------------------------------------- D4e: LightGBM
SMALL_SPACE = dict(n_estimators=(30, 80), learning_rate=(0.05, 0.2), num_leaves=(3, 15), min_child_samples=(8, 20))


@pytest.fixture(scope="module")
def tuned(result):
    return ML.tune_lgbm_per_account(result["panel"], search_space=SMALL_SPACE, n_trials=5, n_inner_splits=2)


@pytest.fixture(scope="module")
def detail_l(result, tuned):
    return ML.cv_lgbm(result["panel"], tuned_params=tuned[0])


def test_outer_cv_first_test_start_matches_the_split(result):
    T = result["panel"]["date"].nunique()
    assert ML.outer_cv_first_test_start(T) == T - cfg.TSCV_N_SPLITS * cfg.TSCV_TEST_SIZE


def test_lgbm_uses_the_same_folds_as_the_other_models(result, detail_l):
    squad = D.squad_pipeline(result["accounts"], ingest.load_raw_balances())
    base, _ = tscv.cv_level_forecasts(result["panel"], squad["balances"])
    keys = ["fold", "origen", "account_id", "h", "date"]
    a = base[base["model"] == "ultimo_saldo"][keys].reset_index(drop=True)
    b = detail_l[keys].reset_index(drop=True)
    assert a.equals(b)
    assert len(detail_l) == cfg.TSCV_N_SPLITS * cfg.TSCV_TEST_SIZE * 6


def test_lgbm_forecast_has_no_lookahead(result, tuned):
    """Alterar TODO lo que ocurre después del origen del corte 1 no puede cambiar sus pronósticos (igual que XGBoost)."""
    panel = result["panel"].copy()
    detail_l1 = ML.cv_lgbm(panel, tuned_params=tuned[0])
    origin = detail_l1[detail_l1["fold"] == 1]["origen"].iloc[0]
    future = panel["date"] > origin
    rng = np.random.default_rng(0)
    for col in ("balance", "inflow", "outflow", "net_flow"):
        panel.loc[future, col] = panel.loc[future, col] * rng.uniform(0.2, 5.0, future.sum())
    altered = ML.cv_lgbm(panel, tuned_params=tuned[0])
    f1 = lambda d: d[d["fold"] == 1].sort_values(["account_id", "h"])["forecast"].to_numpy()
    assert np.allclose(f1(detail_l1), f1(altered))                      # corte 1: idéntico
    assert not np.allclose(detail_l1[detail_l1["fold"] == 5]["forecast"].to_numpy(),
                           altered[altered["fold"] == 5]["forecast"].to_numpy())  # el control: en otros cortes el futuro sí cuenta


def test_cv_lgbm_actually_uses_the_tuned_params_per_account(result):
    """Pasar hiperparámetros distintos para ACC-001 cambia su pronóstico y no el de las demás cuentas."""
    default = ML.cv_lgbm(result["panel"])
    tuned_one = {"ACC-001": dict(cfg.LGBM_BASE_PARAMS, n_estimators=5, learning_rate=0.5, num_leaves=3,
                                 min_child_samples=5, reg_lambda=1.0)}
    changed = ML.cv_lgbm(result["panel"], tuned_params=tuned_one)
    f = lambda d, acc: d[d["account_id"] == acc].sort_values(["fold", "h"])["forecast"].to_numpy()
    assert not np.allclose(f(default, "ACC-001"), f(changed, "ACC-001"))
    assert np.allclose(f(default, "ACC-002"), f(changed, "ACC-002"))


def test_tune_lgbm_never_sees_the_outer_test_folds(result):
    """La garantía central de D4e: alterar los datos EN o DESPUÉS del primer día de prueba de la CV externa no
    puede cambiar ni los parámetros elegidos ni el score con que se eligieron — la afinación no los vio."""
    panel = result["panel"].copy()
    T = panel["date"].nunique()
    cut_day = sorted(panel["date"].unique())[ML.outer_cv_first_test_start(T)]
    future = panel["date"] >= cut_day
    assert future.any()
    rng = np.random.default_rng(0)
    for col in ("balance", "inflow", "outflow", "net_flow"):
        panel.loc[future, col] = panel.loc[future, col] * rng.uniform(0.2, 5.0, future.sum())
    best_before, search_before = ML.tune_lgbm_per_account(result["panel"], search_space=SMALL_SPACE, n_trials=5, n_inner_splits=2)
    best_after, search_after = ML.tune_lgbm_per_account(panel, search_space=SMALL_SPACE, n_trials=5, n_inner_splits=2)
    assert best_before == best_after
    pd.testing.assert_frame_equal(search_before, search_after)


def test_lgbm_pooled_is_close_to_xgboost_but_per_account_tuning_is_not_free_lunch(result, detail_l, detail_x):
    """El hallazgo honesto de D4e: LightGBM como algoritmo (pooled, sin afinar) empata con XGBoost; partir en un
    modelo por cuenta —incluso afinado— pierde más de lo que gana por tener 1/6 de los datos por modelo."""
    pooled = ML.cv_lgbm_pooled(result["panel"])
    all_detail = pd.concat([detail_x, pooled, detail_l], ignore_index=True)
    err = ML.error_table(all_detail, result["panel"])
    assert abs(err.loc["lightgbm_pooled", "MAPE %"] - err.loc["xgboost", "MAPE %"]) < 1.0
    assert err.loc["lightgbm", "MAPE %"] > err.loc["lightgbm_pooled", "MAPE %"]
