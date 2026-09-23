# Enfoque analítico: forecasting por campeonato + motor de optimización

**La pregunta:** cuándo, cuánto y desde dónde mover dinero entre 6 cuentas en 3 monedas para minimizar
el riesgo de faltante sin transferir de más.

**La solución son dos motores, uno montado sobre el otro:**
1. Un **campeonato de modelos de forecasting** que compite y elige, con reglas objetivas, qué pronóstico
   probabilístico del flujo neto usar.
2. Un **ejercicio de optimización** (política de cuantil + MILP) que toma ese pronóstico y decide la
   transferencia óptima.

Nada se diseñó antes de tener evidencia: primero se diagnostica con datos, después se corrige, y solo
entonces se construye.

## Paso 0 · Diagnóstico basado en evidencia, no en intuición
Se reproduce el notebook del squad *tal cual*, sobre los datos crudos — no se asume el defecto, se corre
y se mide (`D1`–`D10`). Hallazgo raíz: pierde 17,6% de las filas, trata 9 etiquetas de moneda como 9
monedas, pronostica el **nivel** del saldo (no el flujo) con una media móvil sin distribución, y "valida"
sumando COP+USD+MXN en un solo número.

## Paso 1 · Datos limpios con una sola regla: la identidad contable
`balance[t] = balance[t-1] + inflow[t] − outflow[t]` cuadra al centavo en toda fila sana. Se usa para
elegir el duplicado correcto, reconstruir nulos y detectar errores de signo/escala por su firma exacta.
Lo que no cuadra va a cuarentena, marcado — nunca se adivina.

## Paso 2 · El campeonato de forecasting
**Qué compite.** Sobre el **flujo neto** (no el saldo, que es solo su acumulado) corre una escalera de
~9 candidatos: naive estacional, ETS (nivel, nivel+estacional, flujos por separado), SARIMAX, Prophet,
bootstrap empírico, XGBoost con rezagos, LightGBM (pooled y afinado por cuenta con Optuna).

**El árbitro.** `TimeSeriesSplit` walk-forward (ventana expansiva, sin fuga): cada modelo entrena solo
con el pasado y se prueba en bloques de 14 días que nunca preceden al entrenamiento.

**El criterio de victoria no es "quién acierta el punto".** Es quién sirve mejor para dimensionar un
colchón de riesgo: pinball loss y cobertura empírica p5–p95, no solo MAPE/sMAPE. `ets_flujos` gana el
error puntual sobre los últimos 14 días reales (D11), pero es `empirical` (bootstrap) quien gana el
campeonato real: corrido dentro del propio motor de optimización, cambiar el forecast más preciso en
punto por el correcto en distribución le **cuesta al sistema 83–131% más** (D12). El campeón no es el
más exacto en promedio — es el mejor calibrado en la cola, que es lo único que le importa a la decisión.

**Es, en esencia, un esquema campeón-retador** — el mismo patrón que la Parte 5 propone para
producción (Databricks + MLflow: cada modelo nuevo entra como `@challenger`, compite en sombra contra
el `@champion`, y solo lo reemplaza si gana con significancia estadística).

## Paso 3 · El ejercicio de optimización
Con el forecast ganador (`empirical`, con su distribución p5–p95) se construyen dos capas de decisión:

1. **Política de cuantil** — dispara una transferencia cuando un cuantil bajo del saldo proyectado cruza
   el piso operativo. Simple, explicable a Tesorería en una frase.
2. **MILP a horizonte rodante (14 días)** — un programa entero-mixto, resuelto **cada día**, que minimiza
   `fee + costo fijo + penalización esperada del déficit`, decidiendo explícitamente cuándo es más barato
   pagar la multa que mover dinero.

Las dos capas dependen de la misma pregunta de fondo: no "cuál es el saldo esperado" sino "qué tan mal
puede ir" — por eso necesitan la distribución del campeonato, no un número suelto.

## Paso 4 · Backtest: forecast + optimización, juntos, contra la historia
La regla del squad, la política de cuantil y el MILP corren sobre el mismo histórico (210 días × 6
cuentas) y el mismo forecast, medidos por moneda:

| Política | Días-cuenta en faltante | Costo (USD) | Transferencias |
|---|---|---|---|
| Sin intervenir | 33 | 0 | 0 |
| Regla del squad | 86 (peor que nada) | 6.275 | 200 |
| Política de cuantil | 0 | 4.955 | 49 |
| **MILP** | **0** | **2.099** | **22** |

El MILP además *decide no transferir* 324 veces, cuando pagar la penalización sale más barato que mover
el dinero — y aun así cierra en 0 faltantes.

## Paso 5 · Estrés antes de confiar
Robusto a FX ±10% y a lag forzado; se fragiliza si el forecast trae sesgo optimista (reaparecen 1–3
días-cuenta en faltante); con un piso de 30 días el problema deja de ser de rebalanceo y pasa a ser de
fondeo — ninguna optimización lo arregla.

## Paso 6 · De vuelta a producción: el mismo campeonato, corriendo solo (vista a futuro)
El despliegue no es un capítulo aparte: es el campeonato del Paso 2 automatizado en un ciclo continuo —
cada reentrenamiento entra como `@challenger`, anota en modo sombra sin tocar el plan real, y solo
reemplaza al `@champion` si gana con significancia estadística. El motor de optimización del Paso 3 sigue
corriendo en modo *dry-run* sobre el `@champion` vigente, con aprobación humana antes de ejecutar
cualquier transferencia.
