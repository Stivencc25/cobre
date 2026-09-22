"""D4d/D4e — Modelos de árboles con rezagos para el saldo a h días, validados con ``TimeSeriesSplit`` y
comparados por MAE / MAPE / sMAPE.

Ambos comparten variables y estrategia directa (D4d los documenta en detalle): una fila por (cuenta, origen *t*,
horizonte *h*), objetivo ``(saldo[t+h] - saldo[t]) / escala[t]`` (el cambio del saldo en días de egreso, ``escala`` =
egreso medio de 28 días de la cuenta — así COP, MXN y USD son comparables). El pronóstico del saldo es
``saldo[t] + ŷ · escala[t]``. Sin fuga: las variables en *t* solo usan datos hasta *t*, y en cada corte el modelo solo
entrena con objetivos que caen dentro del entrenamiento (``t + h <= fin de entrenamiento``).

* **D4d — `cv_xgb`:** un solo modelo XGBoost **global** (las 6 cuentas juntas), hiperparámetros fijos (no se afinan
  sobre los cortes de prueba: `cfg.XGB_PARAMS` lo declara).
* **D4e — `cv_lgbm` / `tune_lgbm_per_account`:** el reto al resultado de D4d — un modelo LightGBM **por cuenta**
  (una serie, un modelo, con sus propios hiperparámetros) en vez de uno global. Aquí sí se afina —con Optuna
  (TPE), no a mano— pero con una CV interna que termina antes de que arranque la CV externa que reporta D4d/D4e
  (ver el docstring de `tune_lgbm_per_account`), así que la afinación no puede inflar el resultado reportado.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from . import config as cfg

MODEL = "xgboost"
MODEL_LGBM = "lightgbm"


def _account_frames(panel: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Por cuenta: variables en cada día *t* (normalizadas por la escala), saldo, escala y día de la semana."""
    out = {}
    reserve = panel.drop_duplicates("account_id").set_index("account_id")["account_type"].eq("reserve")
    for acc, g in panel.sort_values("date").groupby("account_id"):
        g = g.set_index("date")
        scale = g["outflow"].rolling(cfg.OUTFLOW_WINDOW_DAYS, min_periods=cfg.OUTFLOW_MIN_PERIODS).mean()
        cols = {}
        for k in range(cfg.XGB_LAGS):
            cols[f"neto_lag{k}"] = g["net_flow"].shift(k) / scale
        for k in (0, 7):
            cols[f"entrada_lag{k}"] = g["inflow"].shift(k) / scale
            cols[f"salida_lag{k}"] = g["outflow"].shift(k) / scale
        for w in (7, 14, 28):
            cols[f"neto_media{w}"] = g["net_flow"].rolling(w, min_periods=w).mean() / scale
        cols["neto_std14"] = g["net_flow"].rolling(14, min_periods=14).std() / scale
        cols["cobertura_dias"] = g["balance"] / scale
        cols["dow_origen"] = pd.Series(g.index.dayofweek, index=g.index).astype(float)
        cols["es_reserva"] = pd.Series(float(reserve[acc]), index=g.index)
        F = pd.DataFrame(cols)
        out[acc] = {"X": F.to_numpy(float), "names": list(F.columns), "balance": g["balance"].to_numpy(float),
                    "scale": scale.to_numpy(float), "dow": g.index.dayofweek.to_numpy(), "dates": g.index}
    return out


def _stack(frames: dict, origins: dict[str, np.ndarray], H: int, with_target_upto: int | None):
    """Filas (cuenta, origen, h). Con ``with_target_upto`` solo entran los objetivos que caen <= esa posición."""
    X, y, meta = [], [], []
    for acc, f in frames.items():
        for h in range(1, H + 1):
            t = origins[acc]
            if with_target_upto is not None:
                t = t[t + h <= with_target_upto]
            t = t[np.isfinite(f["X"][t]).all(axis=1) & np.isfinite(f["scale"][t])]
            if not len(t):
                continue
            dow_target = f["dow"][np.minimum(t + h, len(f["dow"]) - 1)] if with_target_upto is not None else (f["dow"][t] + h) % 7
            X.append(np.column_stack([f["X"][t], np.full(len(t), h), dow_target]))
            if with_target_upto is not None:
                y.append((f["balance"][t + h] - f["balance"][t]) / f["scale"][t])
            meta.append(pd.DataFrame({"account_id": acc, "t": t, "h": h}))
    return np.vstack(X), (np.concatenate(y) if y else None), pd.concat(meta, ignore_index=True)


def cv_xgb(panel: pd.DataFrame, n_splits: int = cfg.TSCV_N_SPLITS, test_size: int = cfg.TSCV_TEST_SIZE,
           gap: int = cfg.TSCV_GAP, params: dict | None = None, model_name: str = MODEL) -> pd.DataFrame:
    """Pronóstico del saldo con XGBoost en cada corte de ``TimeSeriesSplit``. Mismo formato que ``tscv.cv_level_forecasts``."""
    params = {**cfg.XGB_PARAMS, **(params or {})}
    frames = _account_frames(panel)
    any_acc = next(iter(frames.values()))
    dates = any_acc["dates"]
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(dates), start=1):
        e = int(train[-1])
        steps = (test - e).astype(int)
        H = int(steps.max())
        origins = {a: np.arange(len(f["balance"])) for a, f in frames.items()}
        Xtr, ytr, _ = _stack(frames, origins, H, with_target_upto=e)
        model = XGBRegressor(**params).fit(Xtr, ytr)
        for acc, f in frames.items():
            Xte = np.vstack([np.column_stack([f["X"][[e]], np.array([[h]]), np.array([[(f["dow"][e] + h) % 7]])]) for h in steps])
            yhat = model.predict(Xte)
            fc = f["balance"][e] + yhat * f["scale"][e]
            actual = f["balance"][test]
            rows.append(pd.DataFrame({"fold": fold, "origen": dates[e], "account_id": acc, "model": model_name, "h": steps,
                                      "date": dates[test], "forecast": fc, "real": actual, "error": fc - actual}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------- D4e: modelo por cuenta
def _direct_matrix(f: dict, e: int, steps: np.ndarray) -> np.ndarray:
    """Fila de variables en el origen ``e``, una por cada paso de ``steps`` (más horizonte y día de destino).

    Réplica del bloque de prueba que arma ``cv_xgb`` en línea, factorizada aquí porque ``cv_lgbm`` y
    ``tune_lgbm_per_account`` la necesitan las dos. No toca ``cv_xgb``, que se deja intacto.
    """
    return np.vstack([np.column_stack([f["X"][[e]], np.array([[h]]), np.array([[(f["dow"][e] + h) % 7]])]) for h in steps])


def cv_lgbm(panel: pd.DataFrame, tuned_params: dict[str, dict] | None = None, n_splits: int = cfg.TSCV_N_SPLITS,
            test_size: int = cfg.TSCV_TEST_SIZE, gap: int = cfg.TSCV_GAP, model_name: str = MODEL_LGBM) -> pd.DataFrame:
    """Como ``cv_xgb``, pero UN MODELO POR CUENTA en vez de uno global: cada serie puede tener sus propios
    hiperparámetros (``tuned_params[account_id]``; si falta, usa ``cfg.LGBM_PARAMS``). Mismos cortes de
    ``TimeSeriesSplit`` que ``cv_xgb``/``tscv.cv_level_forecasts`` (mismo formato de salida, comparable con
    ``error_table``/``block_wins``)."""
    frames = _account_frames(panel)
    dates = next(iter(frames.values()))["dates"]
    T = len(dates)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(dates), start=1):
        e = int(train[-1])
        steps = (test - e).astype(int)
        H = int(steps.max())
        for acc, f in frames.items():
            params = {**cfg.LGBM_PARAMS, **(tuned_params or {}).get(acc, {})}
            sub, origins = {acc: f}, {acc: np.arange(T)}
            Xtr, ytr, _ = _stack(sub, origins, H, with_target_upto=e)
            model = LGBMRegressor(**params).fit(Xtr, ytr)
            yhat = model.predict(_direct_matrix(f, e, steps))
            fc = f["balance"][e] + yhat * f["scale"][e]
            actual = f["balance"][test]
            rows.append(pd.DataFrame({"fold": fold, "origen": dates[e], "account_id": acc, "model": model_name, "h": steps,
                                      "date": dates[test], "forecast": fc, "real": actual, "error": fc - actual}))
    return pd.concat(rows, ignore_index=True)


def cv_lgbm_pooled(panel: pd.DataFrame, params: dict | None = None, n_splits: int = cfg.TSCV_N_SPLITS,
                    test_size: int = cfg.TSCV_TEST_SIZE, gap: int = cfg.TSCV_GAP,
                    model_name: str = "lightgbm_pooled") -> pd.DataFrame:
    """Control para D4e: LightGBM GLOBAL (las 6 cuentas juntas, como ``cv_xgb``), con `cfg.LGBM_PARAMS` SIN afinar.

    Aísla si un resultado peor en ``cv_lgbm`` viene del algoritmo o de partir los datos en 6 modelos: aquí se
    cambia solo el algoritmo (LightGBM en vez de XGBoost) y se deja todo lo demás —un modelo, toda la historia,
    hiperparámetros por defecto sin afinar— igual que D4d."""
    frames = _account_frames(panel)
    dates = next(iter(frames.values()))["dates"]
    T = len(dates)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(dates), start=1):
        e = int(train[-1])
        steps = (test - e).astype(int)
        H = int(steps.max())
        origins = {a: np.arange(T) for a in frames}
        Xtr, ytr, _ = _stack(frames, origins, H, with_target_upto=e)
        model = LGBMRegressor(**{**cfg.LGBM_PARAMS, **(params or {})}).fit(Xtr, ytr)
        for acc, f in frames.items():
            yhat = model.predict(_direct_matrix(f, e, steps))
            fc = f["balance"][e] + yhat * f["scale"][e]
            actual = f["balance"][test]
            rows.append(pd.DataFrame({"fold": fold, "origen": dates[e], "account_id": acc, "model": model_name, "h": steps,
                                      "date": dates[test], "forecast": fc, "real": actual, "error": fc - actual}))
    return pd.concat(rows, ignore_index=True)


def outer_cv_first_test_start(T: int, n_splits: int = cfg.TSCV_N_SPLITS, test_size: int = cfg.TSCV_TEST_SIZE,
                               gap: int = cfg.TSCV_GAP) -> int:
    """Primer índice de prueba de la CV externa (día 200 de 270 con los valores por defecto): la frontera que
    ``tune_lgbm_per_account`` no puede cruzar."""
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    train0, _ = next(iter(splitter.split(np.arange(T))))
    return int(train0[-1]) + 1


def _suggest(trial: "optuna.Trial", name: str, spec: tuple) -> float | int:
    """Traduce una entrada de ``cfg.LGBM_SEARCH_SPACE`` (``(lo, hi)`` o ``(lo, hi, "log")``) a un ``trial.suggest_*``."""
    lo, hi, *rest = spec
    log = "log" in rest
    if isinstance(lo, int) and isinstance(hi, int):
        return trial.suggest_int(name, lo, hi, log=log)
    return trial.suggest_float(name, lo, hi, log=log)


def tune_lgbm_per_account(
    panel: pd.DataFrame, search_space: dict | None = None, n_trials: int = cfg.LGBM_TUNE_N_TRIALS,
    n_inner_splits: int = cfg.LGBM_TUNE_N_SPLITS, n_splits: int = cfg.TSCV_N_SPLITS,
    test_size: int = cfg.TSCV_TEST_SIZE, gap: int = cfg.TSCV_GAP, seed: int = cfg.BOOTSTRAP_SEED,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Elige hiperparámetros de LightGBM por cuenta con **Optuna** (TPE) sobre una CV interna ESTRICTAMENTE
    anterior a la CV externa.

    La CV externa (mismos ``n_splits``/``test_size``/``gap`` que ``cv_xgb``) empieza a probar en
    ``outer_cv_first_test_start`` (día 200 con los valores por defecto). La afinación interna solo usa orígenes
    con índice < ese corte (``TimeSeriesSplit`` de nuevo, hacia atrás desde ahí), así que ningún parámetro elegido
    aquí pudo haber visto ni de refilón ninguno de los bloques de prueba que reporta D4e: el resultado no está
    inflado por la afinación, es una garantía de diseño, no una promesa. Un estudio de Optuna por cuenta, sembrado
    (``seed``) y con ``n_jobs=1`` en LightGBM: la búsqueda es reproducible, no solo "mejor que a mano".

    Objetivo de cada intento: MAE en el espacio ya normalizado (días de egreso, el mismo en que se entrena),
    promediado en los cortes internos y los horizontes — comparable entre cuentas en distinta moneda.
    Devuelve ``(mejores_parametros_por_cuenta, tabla_de_intentos)`` para auditar qué se probó y qué ganó.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frames = _account_frames(panel)
    T = len(next(iter(frames.values()))["balance"])
    outer_cut = outer_cv_first_test_start(T, n_splits, test_size, gap)
    space = search_space or cfg.LGBM_SEARCH_SPACE
    inner = TimeSeriesSplit(n_splits=n_inner_splits, test_size=test_size, gap=gap)
    idx = np.arange(outer_cut)                                    # estrictamente antes de la CV externa
    best, rows = {}, []
    for acc, f in frames.items():
        sub, origins = {acc: f}, {acc: np.arange(T)}

        def objective(trial: "optuna.Trial", sub=sub, origins=origins, f=f) -> float:
            params = {**cfg.LGBM_BASE_PARAMS, **{name: _suggest(trial, name, spec) for name, spec in space.items()}}
            errs = []
            for train, test in inner.split(idx):
                e = int(idx[train[-1]])
                steps = (idx[test] - e).astype(int)
                H = int(steps.max())
                Xtr, ytr, _ = _stack(sub, origins, H, with_target_upto=e)
                if ytr is None or len(ytr) < 20:                  # muy poca historia en este corte interno: se salta
                    continue
                model = LGBMRegressor(**params).fit(Xtr, ytr)
                yhat = model.predict(_direct_matrix(f, e, steps))
                y_true = (f["balance"][idx[test]] - f["balance"][e]) / f["scale"][e]
                errs.append(float(np.mean(np.abs(yhat - y_true))))
            return float(np.mean(errs)) if errs else float("inf")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best[acc] = {**cfg.LGBM_BASE_PARAMS, **study.best_params}
        trials = study.trials_dataframe()[["number", "value", *[f"params_{k}" for k in space]]].rename(
            columns={"value": "mae_norm_interno", **{f"params_{k}": k for k in space}})
        trials.insert(0, "account_id", acc)
        trials["elegido"] = trials["number"] == study.best_trial.number
        rows.append(trials)
    return best, pd.concat(rows, ignore_index=True).sort_values(["account_id", "mae_norm_interno"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- métricas
def error_table(detail: pd.DataFrame, panel: pd.DataFrame, by=("model",), band: float = cfg.BAND_PCT) -> pd.DataFrame:
    """MAE, MAPE, sMAPE y reparto sobreestima / balanceado / subestima.

    * ``MAPE`` = media de |F-A|/|A|; ``sMAPE`` = media de 2|F-A|/(|F|+|A|) (ambos en %).
    * ``rel`` = (F-A)/A: **sobreestima** si rel > +band (riesgo de faltante: el saldo real es menor al esperado),
      **subestima** si rel < -band (movimientos innecesarios), **balanceado** en el resto.
    * ``MAE`` va en días de egreso (|F-A| / egreso medio de 28 días al origen) para poder agrupar cuentas en monedas distintas;
      con ``by`` que incluya ``account_id`` se agrega también el MAE en la moneda de la cuenta.
    """
    outflow = panel.pivot(index="date", columns="account_id", values="outflow").sort_index()
    scale = (outflow.rolling(cfg.OUTFLOW_WINDOW_DAYS, min_periods=cfg.OUTFLOW_MIN_PERIODS).mean()
             .stack().rename("escala").reset_index().rename(columns={"date": "origen"}))
    d = detail.merge(scale, on=["account_id", "origen"], how="left")
    d["rel"] = d["error"] / d["real"].abs()
    d["ape"] = d["rel"].abs() * 100
    d["smape"] = 200 * d["error"].abs() / (d["forecast"].abs() + d["real"].abs())
    d["mae_dias"] = d["error"].abs() / d["escala"]
    d["sobre"] = d["rel"] > band
    d["sub"] = d["rel"] < -band
    d["bal"] = ~(d["sobre"] | d["sub"])
    by = list(by)
    agg = {"MAE (días de egreso)": ("mae_dias", "mean"), "MAPE %": ("ape", "mean"), "sMAPE %": ("smape", "mean"),
           "sesgo % (+ = sobreestima)": ("rel", lambda s: s.mean() * 100),
           f"sobreestima % (>+{band:.0%})": ("sobre", lambda s: s.mean() * 100),
           f"balanceado % (±{band:.0%})": ("bal", lambda s: s.mean() * 100),
           f"subestima % (<-{band:.0%})": ("sub", lambda s: s.mean() * 100), "n": ("error", "size")}
    if "account_id" in by:
        agg = {"MAE (moneda de la cuenta)": ("error", lambda s: s.abs().mean()), **agg}
    return d.groupby(by).agg(**agg)


def feature_importance(panel: pd.DataFrame, top: int = 12) -> pd.Series:
    """Importancia (ganancia) de cada variable en un modelo entrenado con TODO el historial disponible (h = 1..14)."""
    frames = _account_frames(panel)
    H = cfg.DECISION_HORIZON
    origins = {a: np.arange(len(f["balance"])) for a, f in frames.items()}
    last = len(next(iter(frames.values()))["balance"]) - 1
    Xtr, ytr, _ = _stack(frames, origins, H, with_target_upto=last)
    model = XGBRegressor(**{**cfg.XGB_PARAMS, "importance_type": "gain"}).fit(Xtr, ytr)
    names = next(iter(frames.values()))["names"] + ["horizonte_h", "dow_destino"]
    imp = pd.Series(model.feature_importances_, index=names)
    return (imp / imp.sum()).sort_values(ascending=False).head(top)


def block_wins(detail: pd.DataFrame, model: str, rival: str) -> tuple[int, int]:
    """En cuántos bloques (corte × cuenta) ``model`` tiene menor MAE que ``rival``. Devuelve ``(victorias, bloques)``."""
    mae = detail.assign(a=detail["error"].abs()).groupby(["fold", "account_id", "model"])["a"].mean().unstack("model")
    return int((mae[model] < mae[rival]).sum()), len(mae)
