# Liquidez multi-banco — caso técnico (Technical Lead, Data Science)

Desarrollo propio para el take-home case ["Optimizing Multi-Bank Liquidity and Cash Flow"](take_home_case.md):
una fintech mueve dinero entre 6 cuentas en 3 monedas (COP, USD, MXN) y necesita decidir, de forma
proactiva, **cuándo, cuánto y desde dónde** rebalancear para no quedarse corta de liquidez sin mover
dinero de más. Un squad hizo un primer intento (`data/raw/flawed_model_squad_draft.ipynb`) que "corre sin
error" pero está roto en varios niveles; este repo lo diagnostica, lo corrige y propone un enfoque nuevo.

**Entregable principal:** [`notebooks/flawed_model_squad_draft_v4_es.ipynb`](notebooks/flawed_model_squad_draft_v4_es.ipynb)
(también disponible en inglés: [`flawed_model_squad_draft_v4_en.ipynb`](notebooks/flawed_model_squad_draft_v4_en.ipynb)).

**Documentos complementarios**, para quien prefiera leer antes de abrir el notebook:
- [`ENFOQUE_ANALITICO.md`](ENFOQUE_ANALITICO.md) — resumen corto (ES) del método: campeonato de forecasting +
  motor de optimización, y por qué están en ese orden.
- [`DOCUMENTATION.md`](DOCUMENTATION.md) — walkthrough largo (EN), paso a paso, con una figura por paso y los
  números citados directamente del notebook.

## El proceso, en orden

1. **Reproducir el notebook del squad tal cual**, sobre los datos crudos, para tener evidencia — no se
   asume el defecto, se corre su código y se mide el efecto (Parte 1, `D1`–`D10`).
2. **Corregir los datos con la identidad contable** `balance[t] = balance[t-1] + inflow[t] − outflow[t]`,
   que cuadra al centavo en toda fila sana. Sirve para elegir el duplicado correcto entre dos versiones de
   un mismo día, reconstruir nulos y detectar errores de signo/escala con firma exacta. Lo que ninguna
   regla explica va a cuarentena marcada, no se adivina (`src/ingest.py`, `src/quality.py`).
3. **Rehacer el forecast sobre el flujo neto, no sobre el nivel del saldo**, con un campeonato de ~9 modelos
   de más simple a más complejo (naive estacional → ETS → SARIMAX → Prophet → bootstrap empírico → XGBoost →
   LightGBM pooled/afinado por cuenta), validado con `TimeSeriesSplit` walk-forward sin fuga de información
   (`src/forecast.py`, `src/ml_forecast.py`, `src/tscv.py`). El criterio de victoria no es acertar el punto,
   es servir mejor para dimensionar un colchón de riesgo: gana el bootstrap empírico por pinball loss y
   cobertura p5–p95, aunque `ets_flujos` le gane en error puntual sobre los últimos 14 días reales — corrido
   dentro del propio motor, cambiar el forecast más preciso en punto por el correcto en distribución le
   cuesta al sistema 83–131% más (D11/D12).
4. **Diseñar el motor de rebalanceo en dos capas sobre ese mismo forecast** (`src/policy.py`,
   `src/optimize.py`): una política de cuantil, explicable a Tesorería, y un MILP a horizonte rodante de
   14 días que minimiza `fee + costo fijo + penalización esperada del déficit`, reoptimizando cada día y
   decidiendo explícitamente cuándo es más barato pagar la penalización que transferir.
5. **Backtest con contrafactual**: la regla del squad, la política de cuantil y el MILP corren sobre el
   mismo histórico y el mismo forecast, medidos por moneda (nunca sumados) — días-cuenta en faltante,
   costo total, número de transferencias (`src/backtest.py`).
6. **Explicar el resultado a dos audiencias** (data scientist del squad / Tesorería), documentar dónde se
   usó IA y dónde no, y describir cómo llevarlo a producción — Partes 3, 4 y 5 del notebook.

Todos los supuestos de negocio que no son observables en los datos (piso operativo, TRM, costo fijo,
penalización) viven en un solo lugar, `src/config.py`, marcados `[DECISIÓN HUMANA]` donde son un valor por
defecto pendiente de validar con Tesorería — nunca una constante suelta en el código de modelado.

## Resumen ejecutivo

- **El "parece funcionar" del squad es falso.** Su pipeline pierde 17,6% de las filas por un `dropna`
  silencioso tras `coerce`, trata 9 etiquetas de moneda como 9 monedas, elige el donante por saldo nominal
  sin convertir FX, y "valida" con un promedio antes/después que suma COP+USD+MXN (−9,6% presentado como
  éxito). Aplicada sobre el histórico completo, su regla deja **86 días-cuenta en faltante — peor que no
  hacer nada (33)**.
- **Datos corregidos con la identidad contable:** panel 6 cuentas × 270 días, 0 nulos, 0 negativos, residuo
  ≈ 0,01. Quedan 6 episodios de 3 días sin explicar, en cuarentena.
- **Forecast probabilístico del flujo neto:** todos los modelos le ganan 26–34% al naive estacional, pero
  **ninguno le gana a un bootstrap empírico** — el valor está en la distribución (cobertura p5–p95 ≈ 93%),
  no en el punto. `ets_flujos` gana en error puntual sobre los últimos 14 días reales (D11), pero usarlo en
  el motor en vez del bootstrap empírico sube el costo del backtest 83–131% (D12): el modelo más exacto en
  promedio no es el mejor calibrado en la cola, que es lo único que le importa a la decisión.
- **Backtest (210 días × 6 cuentas, piso = 25 días de egresos):**

  | Política | Días-cuenta en faltante | Costo (USD) | Transferencias |
  |---|---|---|---|
  | Sin intervenir | 33 | 0 | 0 |
  | Regla del squad | 86 (peor que nada) | 6.275 | 200 |
  | Política de cuantil | 0 | 4.955 | 49 |
  | **MILP** | **0** | **2.099** | **22** (todas misma moneda) |

  El MILP también *decide no transferir* cuando el costo de operar supera la penalización esperada
  (324 de esas decisiones en el backtest, 0 déficit ex-post).
- **Sensibilidad:** robusto a FX ±10% y a lag forzado; frágil a un forecast con sesgo optimista (aparecen
  1–3 días-cuenta en faltante); con un piso de 30 días el sistema deja de tener liquidez suficiente — ahí
  el problema es de fondeo, no de rebalanceo.
- **Pendiente de decisión humana:** el piso operativo (K), la TRM oficial del COP (hoy 3.900, aproximada),
  el costo fijo y la penalización por déficit, los 6 episodios sin explicar, y qué representan las 140
  transferencias del log (no aparecen en los saldos).

## Estado del entregable

La Parte 1 (`D1`–`D12`), el núcleo técnico de la Parte 2 (forecast, políticas, motor de rebalanceo) y las
Partes 3, 4 y 5 están completos, en ambas versiones del notebook (`_es`/`_en`). El razonamiento de negocio
detrás de cada paso está además en prosa en [`ENFOQUE_ANALITICO.md`](ENFOQUE_ANALITICO.md) y
[`DOCUMENTATION.md`](DOCUMENTATION.md).

## Estructura
```
data/raw/         originales inmutables: accounts.csv, account_balances_daily_RAW.csv,
                  transfers_log_RAW.csv y flawed_model_squad_draft.ipynb (el notebook del squad, evidencia)
data/processed/   generado por `python -m src.ingest` (panel.csv, transfers_clean.csv); en .gitignore
notebooks/        flawed_model_squad_draft_v4_es.ipynb  <- el entregable (importa de src/, no define lógica)
                  flawed_model_squad_draft_v4_en.ipynb  <- misma versión, en inglés
DOCUMENTATION.md  walkthrough largo (EN) del proceso completo, con figuras
ENFOQUE_ANALITICO.md  resumen corto (ES) del método
src/config.py     supuestos y constantes de negocio (única fuente; [DECISIÓN HUMANA] marcadas)
src/ingest.py     F1  ingesta y reconciliación contable
src/quality.py    F1  reporte de calidad
src/features.py   F2  features y reconciliación con el log de transferencias
src/forecast.py   F3  escalera de modelos base, bootstrap, backtest walk-forward
src/seasonality.py, src/tscv.py   F3  estacionariedad/estacionalidad (D4c) y el arnés de walk-forward CV
src/ml_forecast.py                F3  candidatos ML del campeonato: XGBoost, LightGBM pooled/afinado (D11)
src/policy.py     F4  estado, política de cuantil, regla del squad
src/optimize.py   F4  MILP (scipy/HiGHS), horizonte rodante
src/backtest.py   F5  simulador, métricas por moneda, frontera, sensibilidad
src/diagnosis.py  Parte 1: evidencia de D1–D10
src/plots.py      figuras -> reports/figures/
tests/            59 tests (pytest)
```

## Comandos
```
python -m pytest              # tests
python -m src.ingest          # regenera data/processed/ y muestra el reporte de calidad
jupyter nbconvert --to notebook --execute --inplace notebooks/flawed_model_squad_draft_v4_es.ipynb   # ~5-6 min
```
