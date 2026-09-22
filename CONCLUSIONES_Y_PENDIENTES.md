# Conclusiones y pendientes — caso de liquidez multi-banco

Entregable: [`notebooks/flawed_model_squad_draft_.ipynb`](notebooks/flawed_model_squad_draft_.ipynb) (ejecutado, sin errores) + `src/` + `tests/` (28 tests en verde).
Cifras del run final; la última celda del notebook verifica 10 de ellas automáticamente.

---

## 1. Conclusiones

### El diagnóstico del squad
- **"Parece funcionar" es falso.** Su validación (promedio de saldos antes/después del 1-mayo) suma COP+USD+MXN, no tiene contrafactual, y su propia celda imprime −9,6% como éxito.
- **Pierde datos y no lo dice:** 293 de 1.668 filas (17,6%) por `coerce` + `dropna`; 9 etiquetas de moneda tratadas como 9 monedas.
- **Su forecast es peor que "el saldo de mañana = el de hoy"** (error ~23% mayor, siempre sobreestima porque los saldos derivan a la baja) y no tiene incertidumbre.
- **Elige el donante por saldo nominal sin convertir moneda** y gira el monto en la moneda equivocada. Los umbrales son fijos, solo para 4 de 6 cuentas; el de ACC-004 está por encima de su propia mediana, así que dispara ~72% de los días por construcción.

### Los datos
- La **identidad contable** `b[t] = b[t-1] + in − out` cuadra al centavo en toda fila sana. Con ella se elige el duplicado correcto, se reconstruyen nulos y se reparan errores con firma exacta.
- Panel final: 6 cuentas × 270 días, 0 nulos, 0 negativos.

### Correcciones al spec (los datos lo contradicen)
| Spec decía | Los datos muestran |
|---|---|
| 4 balances negativos por sign flip | 7 errores puntuales (4 signos + 3 decimales ×10) **más 6 episodios de 3 días** que ninguna regla explica → cuarentena |
| Residuos de la identidad se explican por transferencias liquidadas | **0 residuos.** Las transferencias no están en los saldos (las de USD son 4,7× el mayor flujo diario) |
| Lag mediana 2 y fee 0,23% (un solo número) | **Dos regímenes:** misma moneda 0,04% y 0–1 día; cross-moneda ~0,27% y 1–3 días |
| Sin `transfer_id` duplicados | Cierto, pero hay **5 transferencias duplicadas** con otro id |

### El forecast
- Todos los modelos ganan 26–34% al naive estacional, pero **ninguno gana a un bootstrap empírico**: la media del flujo neto casi no tiene señal. El valor está en la **distribución** (cobertura p5–p95 ≈ 93%), no en el punto.

### El backtest (210 días × 6 cuentas, piso = 25 días de egresos)
| Política | Días-cuenta en faltante | Costo (USD) | Transferencias |
|---|---|---|---|
| Sin hacer nada | 33 | 0 | 0 |
| Regla del squad | **86** (peor que nada) | 6.275 | 200 |
| Cuantil p5 | 0 | 4.955 | 49 |
| **MILP** | **0** | **2.099** | **22 (todas misma moneda)** |

- La regla del squad queda **dominada** en la frontera costo–riesgo. Corregir solo el FX no la salva (36 días en faltante).
- El MILP **decide no transferir** cuando la penalización esperada es menor que el costo de operar (324 decisiones de este tipo, 0 déficit ex-post).
- Subir λ más allá de la rodilla solo compra costo sin bajar el déficit.

### Sensibilidad
- **Robusto** a FX ±10% y a lag forzado a 3 días.
- **Frágil a un forecast sesgado:** con sesgo optimista de +0,5σ aparecen 1 (cuantil) y 3 (MILP) días en faltante.
- **Con piso de 30 días el problema deja de ser de rebalanceo:** el excedente agregado del sistema baja a ~USD 58 mil. Es un problema de fondeo.

---

## 2. Tus pendientes

### A. Antes de entregar (bloqueantes)

- [ ] **Revisar la Parte 4 (uso de IA).** Es tu declaración ante quien evalúa. Está escrita con lo que pasó en esta sesión; ajústala a tu experiencia real y a lo que quieras decir que hiciste tú.
- [ ] **Decidir el idioma.** El notebook está en español; el enunciado original está en inglés. Si la entrega va en inglés, hay que traducirlo.
- [ ] **Validar los supuestos [DECISIÓN HUMANA]** (todos en `src/config.py`). Los puse yo como valores por defecto:
  - [ ] **TRM del COP:** hoy 3.900, *aproximado y no verificado*. Sustituir por la TRM oficial (Banco de la República).
  - [ ] **Piso operativo (K = 25 días de egresos).** Define qué es un faltante. Elegido por solvencia agregada; confirma que Tesorería lo acepta.
  - [ ] **Costo fijo por operación (USD 25)** y **penalización λ (0,2% del déficit por cuenta-día).** λ se fijó viendo la misma historia que se evalúa (en muestra).
  - Si cambias alguno: reejecuta el notebook (~6 min) y las cifras y textos generados se actualizan. Los textos escritos a mano de las Partes 4 y 5 no.

### B. Decisiones abiertas para plantear (están al final del notebook)

- [ ] **Los 6 episodios de 3 días** (caídas de 65–85% con rebote): ¿eventos reales o errores de registro? Si son reales, el riesgo de faltante está subestimado.
- [ ] **¿Qué representan las 140 transferencias del log,** si no aparecen en los saldos?
- [ ] **Duplicados de balance:** solo una fila de cada par cuadra con la identidad, lo que apunta a una recarga que reescribe el saldo pero no los flujos. Revisar en origen.

### C. Formato de entrega

- [ ] **PDF del write-up.** El spec pide notebook + PDF + repo. El write-up (Partes 3–5) está dentro del notebook; falta exportarlo o generar el PDF.
- [ ] **Repo público en GitHub.** Hoy no es un repo git. Está listo `.gitignore` y `README.md`. Ojo: los CSV crudos y `venv/` no deberían subirse sin decidirlo.
- [x] **Estructura del repo:** hecha. CSV originales en `data/raw/`, notebook en `notebooks/`, rutas en `src/config.py`. Los notebooks exploratorios previos (`data_quality_diagnostics*.ipynb`) siguen en la raíz y **ya no encuentran los CSV** (leen `accounts.csv` etc. desde la carpeta actual): si los vas a usar, ajusta sus rutas a `data/raw/`.

### D. Para tener listo en la entrevista

- **Por qué gana `empirical`:** ETS y SARIMAX mejoran al naive, pero no a la distribución empírica. Es honesto, pero incómodo si preguntan "¿dónde está el modelo?".
- **Limitación de λ:** se eligió mirando la misma muestra; no hay validación fuera de muestra.
- **Supuesto de exogeneidad:** los flujos no se reajustan cuando se mueve saldo entre cuentas (los egresos parecen escalar con el nivel, ~35 días de cobertura).
- **El colchón de sesgo:** el forecast optimista es la sensibilidad que sí duele; por eso el monitoreo de sesgo/cobertura (Parte 5) es un control, no un adorno.
