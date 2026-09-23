"""Supuestos y constantes de negocio del caso de liquidez multi-banco.

Regla del proyecto: NINGUNA constante de negocio vive fuera de este archivo.
Los supuestos marcados [DECISIÓN HUMANA] no son observables en los datos: son
valores por defecto declarados, razonables y fáciles de cambiar. Las tablas de
sensibilidad del backtest existen precisamente para no depender de ellos a ciegas.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"   # originales inmutables: accounts, balances RAW, transfers RAW, notebook del squad
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

ACCOUNTS_FILE = DATA_RAW / "accounts.csv"
BALANCES_FILE = DATA_RAW / "account_balances_daily_RAW.csv"
TRANSFERS_FILE = DATA_RAW / "transfers_log_RAW.csv"
PANEL_FILE = DATA_PROCESSED / "panel.csv"
TRANSFERS_CLEAN_FILE = DATA_PROCESSED / "transfers_clean.csv"

# ------------------------------------------------------------ contrato de datos
EXPECTED_RAW_ROWS = 1668
EXPECTED_ACCOUNTS = 6
EXPECTED_DAYS = 270
EXPECTED_PANEL_ROWS = EXPECTED_ACCOUNTS * EXPECTED_DAYS   # 1.620
DATE_FORMATS = (                       # cascada; los separadores son inequívocos
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),
    (r"\d{2}-\d{2}-\d{4}", "%m-%d-%Y"),
)
CURRENCY_LABEL_MAP = {
    "cop": "COP", "colombian peso": "COP",
    "usd": "USD", "us dollar": "USD",
    "mxn": "MXN", "mexican peso": "MXN",
}
RECONCILIATION_TOL = 1.0               # unidades de moneda; las filas limpias cuadran a <= 0,01
PLAUSIBLE_REL_CHANGE = 0.25            # lectura no verificable pero plausible (±25% del vecino confiable)
SIGN_FLIP_RATIO = -1.0
DECIMAL_SHIFT_RATIO = 10.0
SIGNATURE_RTOL = 1e-3
RECON_MAX_PASSES = 20
WEEKDAY_IMPUTE_WINDOW_WEEKS = 4        # flujos brutos irrecuperables: mediana del mismo día de semana

# ------------------------------------------------------------------------ FX
# Unidades de moneda por 1 USD. [DECISIÓN HUMANA] No hay FX en ningún archivo.
FX_PER_USD = {
    "USD": 1.0,
    "MXN": 18.4457,   
    "COP": 3900.0,    #banrep
}
FX_SOURCE_NOTE = (
    "MXN: FIX Banxico 26-sep-2025 = 18,4457. COP: aproximado ~3.900, pendiente de "
    "sustituir por la TRM oficial (Banco de la República / Superfinanciera)."
)

# ------------------------------------------- transferencias (calibrado del log)
# El pooled del spec (lag mediana 2, fee media 0,23%) mezcla DOS regímenes:
#   misma moneda : fee = 0,04% constante, lag 0-1 días
#   cross-moneda : fee 0,15%-0,40% (media ~0,27%), lag 1-3 días
FEE_RATE = {"same_ccy": 0.0004, "cross_ccy": 0.0027}
LAG_OBSERVED = {                       # {lag_días: frecuencia}; se recalcula y valida en los tests
    "same_ccy": {0: 20, 1: 5},
    "cross_ccy": {1: 37, 2: 57, 3: 26},
}
LAG_PLAN = {"same_ccy": 1, "cross_ccy": 3}     # p90 observado por tipo: se planifica conservador
LAG_STRESS_DAYS = 3                            # escenario de sensibilidad: lag forzado
MIN_TRANSFER_DAYS_OF_OUTFLOW = 1.0             # monto mínimo = 1 día de egresos del destino
FIXED_COST_USD = 25.0                          # [DECISIÓN HUMANA] costo fijo por operación
# [DECISIÓN HUMANA] fracción del déficit por cuenta-día. Punto de operación elegido en la RODILLA de la
# frontera costo-riesgo del backtest (λ mayor solo compra costo sin bajar el déficit: ver notebook); se
# fijó DESPUÉS de ver la frontera, así que es una elección en muestra, no validada fuera de muestra.
DEFICIT_PENALTY_PER_DAY = 0.002
PENALTY_WEIGHT = {"operational": 1.0, "reserve": 0.5}   # operativa protegida antes que reserva
TIEBREAK_EARLY = 1e-7                          # desempate: a igual costo, transferir antes

# ------------------------------------------------- nivel mínimo operativo (piso)
# Faltante := saldo < piso. [DECISIÓN HUMANA] Piso = K días de egresos medios (ventana móvil, sin lookahead).
# K=25 se elige por SOLVENCIA AGREGADA: con K=30 la suma de pisos casi iguala a la liquidez total del
# sistema en el día más ajustado (excedente mínimo ~USD 58 mil): ahí el faltante es de fondeo, no de
# rebalanceo. Con K=25 el excedente agregado mínimo es ~USD 1,5 millones. Se reporta la tabla y K=30 va como sensibilidad.
MIN_COVER_DAYS = {"operational": 25.0, "reserve": 25.0}
OUTFLOW_WINDOW_DAYS = 28
OUTFLOW_MIN_PERIODS = 14

# ------------------------------------------------------------------- forecast
HORIZONS = (7, 14)
DECISION_HORIZON = 14                  # horizonte del LP y de la política de cuantil
MIN_TRAIN_DAYS = 60                    # historia mínima antes del primer origen
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
INTERVAL = (0.05, 0.95)
N_DRAWS_EVAL = 1000                    # trayectorias simuladas para evaluar el forecast
N_SCENARIOS_LP = 40                    # escenarios del LP (SAA)
BOOTSTRAP_SEED = 20250927
MODEL_TOLERANCE = 0.02                 # empate técnico: dentro de 2% del mejor pinball
SEASONAL_PERIOD = 7

# ------------------------------------------------- estacionalidad y validación temporal (D4b)
SEASONAL_ALPHA = 0.01                  # nivel de significancia del test de día de semana (Kruskal-Wallis)
TSCV_N_SPLITS = 5                      # cortes de TimeSeriesSplit (ventana expansiva)
TSCV_TEST_SIZE = DECISION_HORIZON      # días de cada bloque de prueba: el horizonte de decisión
TSCV_GAP = 0                           # días entre entrenamiento y prueba (0: el pronóstico arranca al día siguiente)
ACF_NLAGS = 28                         # rezagos del ACF/PACF (4 semanas)
SEASONAL_LAGS = (7, 14, 21, 28)        # rezagos donde se busca el patrón semanal
ADF_ALPHA = 0.05                       # ADF: si p > alpha no se rechaza raíz unitaria
OVERDIFF_ACF1 = -0.25                  # ACF(1) de la diferencia por debajo de esto = sobrediferenciada (lo teórico es -0,5: se toma el punto medio)
MSTL_PERIODS = (7, 30)                 # semanal + ~mensual, para quitar la tendencia antes de leer el ACF
BAND_PCT = 0.15                        # banda de "balanceado": pronóstico dentro de ±15% del saldo real
XGB_LAGS = 7                           # rezagos diarios del flujo neto (y de entradas/salidas) como variables
XGB_PARAMS = dict(                     # fijos y modestos: ~1.000 orígenes por cuenta no sostienen un modelo grande; NO se afinan sobre los cortes de prueba
    n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=10.0, objective="reg:absoluteerror", tree_method="hist", n_jobs=1, random_state=BOOTSTRAP_SEED,
)
# --------------------------------------------------------- D4e: LightGBM por cuenta, con afinación anidada
# A diferencia de XGB_PARAMS (fijos, sin afinar), aquí SÍ se afina — es el punto del ejercicio D4e: un modelo
# por serie en vez de uno global. Para que siga siendo honesto, la búsqueda usa una CV interna que termina
# ANTES de que arranque la CV externa de D4c/D4d (día 200 con los valores por defecto: ver
# ``ml_forecast.tune_lgbm_per_account``); ningún combo pudo haber visto ninguno de los bloques que reporta D4e.
LGBM_BASE_PARAMS = dict(objective="regression_l1", verbosity=-1, random_state=BOOTSTRAP_SEED, n_jobs=1,
                         subsample_freq=1)   # sin esto, LightGBM ignora `subsample` en silencio (bagging_freq=0 por defecto)
LGBM_PARAMS = dict(                    # combo por defecto si una cuenta no se afina
    **LGBM_BASE_PARAMS, n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
)
LGBM_TUNE_N_SPLITS = 3                  # cortes de la CV interna de afinación (estrictamente antes de la externa)
LGBM_TUNE_N_TRIALS = 30                 # pruebas de Optuna (TPE) por cuenta; datos de sobra son pocos, no vale la pena más
LGBM_SEARCH_SPACE = dict(               # rangos que explora Optuna; los mismos límites conservadores del grid anterior
    n_estimators=(50, 400), learning_rate=(0.01, 0.1, "log"), num_leaves=(3, 31),
    min_child_samples=(8, 40), reg_lambda=(1.0, 30.0, "log"), subsample=(0.6, 1.0), colsample_bytree=(0.6, 1.0),
)

WAIT_MIN_SLACK_DAYS = 1                # política D: se puede esperar solo si aún sobra >= 1 día antes del último día útil para solicitar
WAIT_RECOVERY_MARGIN_DAYS = 1.0        # política D: la mediana debe quedar >= piso + 1 día de egreso en los días que la transferencia puede cubrir
ETS_LEVEL_DAMPED = True                # tendencia amortiguada: evita extrapolar la deriva a horizontes largos

# ------------------------------------------------------ política de cuantil / LP
DONOR_MARGIN_DAYS = 5.0                # política B: el donante conserva piso + 5 días de egresos (histéresis anti ping-pong)
QUANTILE_POLICY_ALPHA = 0.05           # cuantil bajo del saldo proyectado
LP_TIME_LIMIT_S = 20.0
LP_MIP_GAP = 1e-2
LP_MAX_REQUEST_DAYS = 5                # días de solicitud candidatos dentro del horizonte
LP_PRESCREEN_PROB = 0.0                # destinatario candidato si algún escenario incumple el piso

# --------------------------------------------------- regla del squad (evidencia)
SQUAD_THRESHOLD = {
    "ACC-001": 1_000_000_000,
    "ACC-002": 100_000,
    "ACC-004": 40_000_000,
    "ACC-006": 100_000,
}
SQUAD_WINDOW = 14
SQUAD_CUTOVER = "2025-05-01"

# ------------------------------------------------------------------- backtest
BACKTEST_LAG_SEED = 7
IDLE_MARGIN = 0.20                     # saldo ocioso := saldo por encima de piso × (1 + margen)
SENSITIVITY_FX_SHOCK = 0.10
SENSITIVITY_BIAS_SD = 0.5              # sesgo inyectado al forecast, en desv. estándar diarias por día
LP_PENALTY_GRID = (0.0005, 0.002, 0.005, 0.01, 0.03, 0.1)
QUANTILE_ALPHA_GRID = (0.01, 0.05, 0.10, 0.25, 0.50)


def assumptions_table():
    """Tabla de supuestos que se imprime junto a cada plan y cada resultado."""
    import pandas as pd

    rows = [
        ("FX (unid. por USD)", str(FX_PER_USD), "[DECISIÓN HUMANA] " + FX_SOURCE_NOTE),
        ("Fee variable", str(FEE_RATE), "Observado en el log: misma moneda 0,04% fijo; cross 0,15-0,40%"),
        ("Lag observado", str(LAG_OBSERVED), "Observado en el log"),
        ("Lag de planificación", str(LAG_PLAN), "p90 por tipo (conservador)"),
        ("Ponderación penalización", str(PENALTY_WEIGHT), "operativa > reserva como destino"),
        ("Monto mínimo de transferencia", f"{MIN_TRANSFER_DAYS_OF_OUTFLOW} día(s) de egresos del destino",
         "evita goteo de micro-operaciones"),
        ("Cuantil de la política B", f"p{int(QUANTILE_POLICY_ALPHA*100)}", "buffer derivado de la distribución"),
        ("Escenarios LP", str(N_SCENARIOS_LP), "aproximación por muestreo (SAA)"),
        ("Donante política B", f"conserva piso + {DONOR_MARGIN_DAYS:g} días de egresos", "histéresis anti ping-pong"),
    ]
    return pd.DataFrame(rows, columns=["supuesto", "valor", "origen / nota"])
