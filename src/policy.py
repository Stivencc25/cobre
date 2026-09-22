"""F4 (capa 1) — Estado, primitivas de decisión, política de cuantil y regla del squad.

Convenciones (se aplican igual a TODAS las políticas del backtest):
* Decisión al cierre del día ``t`` con información hasta ``t``; la transferencia se SOLICITA el
  día ``t+1``: el origen se debita ese día (monto + fee) y el destino recibe en
  ``t+1+lag`` (monto convertido a FX). Nunca llega antes de lo que permite el lag.
* Toda cifra está en moneda nativa; nada de monedas distintas se compara sin FX de config.
* Faltante := saldo < piso, con  piso = K días × egreso medio de los últimos 28 días (sin lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from .forecast import OriginForecast


# ------------------------------------------------------------------------ Book
@dataclass
class Book:
    """Datos estáticos del problema: cuentas, saldos orgánicos, egresos, pisos y FX de planificación."""

    accounts: list[str]
    ccy: list[str]
    acc_type: list[str]
    dates: pd.DatetimeIndex
    balance: np.ndarray                # (T, A) saldo orgánico (sin transferencias)
    outflow: np.ndarray                # (T, A)
    net_flow: np.ndarray               # (T, A)
    reliable: np.ndarray               # (T, A) bool
    avg_outflow: np.ndarray            # (T, A) egreso medio de los 28 días ANTERIORES a cada día
    floor: np.ndarray                  # (T, A) piso vigente para evaluar cada día
    fx: np.ndarray                     # (A,) unidades por USD (planificación)

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, cover_days: dict[str, float] | None = None,
                    threshold_override: dict[str, float] | None = None) -> "Book":
        """``threshold_override`` (p. ej. ``cfg.SQUAD_THRESHOLD``) reemplaza el piso dinámico por un valor fijo,
        cuenta por cuenta, para las cuentas que tenga — no se descarta, se conserva como opción explícita. Las
        cuentas que no aparezcan en el diccionario (p. ej. las de reserva, sin umbral del squad) se quedan con el
        piso dinámico de siempre (``K × egreso medio de 28 días``).
        """
        piv = lambda c: panel.pivot(index="date", columns="account_id", values=c).sort_index()
        bal, out, net, rel = piv("balance"), piv("outflow"), piv("net_flow"), piv("reliable")
        meta = panel.drop_duplicates("account_id").set_index("account_id").loc[bal.columns]
        avg = out.rolling(cfg.OUTFLOW_WINDOW_DAYS, min_periods=cfg.OUTFLOW_MIN_PERIODS).mean().shift(1)
        k = meta["account_type"].map(cover_days or cfg.MIN_COVER_DAYS).to_numpy(float)
        floor = avg.to_numpy(float) * k
        if threshold_override:
            for acc, tau in threshold_override.items():
                if acc in bal.columns:
                    floor[:, bal.columns.get_loc(acc)] = float(tau)
        return cls(
            accounts=list(bal.columns), ccy=list(meta["currency"]), acc_type=list(meta["account_type"]),
            dates=bal.index, balance=bal.to_numpy(float), outflow=out.to_numpy(float), net_flow=net.to_numpy(float),
            reliable=rel.to_numpy(bool), avg_outflow=avg.to_numpy(float), floor=floor,
            fx=np.array([cfg.FX_PER_USD[c] for c in meta["currency"]]),
        )

    def extended(self, days: int = 1) -> "Book":
        """Copia con ``days`` filas más al final (calendario diario) para decidir en el ÚLTIMO día con datos.

        El piso de un día ``t+1`` depende del egreso medio de los 28 días que terminan en ``t``, así que decidir en el último
        día exige el piso del día siguiente. Las filas nuevas repiten el egreso medio y el piso de esa última ventana; saldo,
        egreso y flujo neto van vacíos (NaN): no hay datos futuros. No sirve para simular, solo para decidir hoy.
        """
        w = cfg.OUTFLOW_WINDOW_DAYS
        avg_last = np.nanmean(self.outflow[-w:], axis=0)
        cover = self.floor[-1] / self.avg_outflow[-1]                       # días de cobertura por cuenta (K)
        pad = lambda a, v: np.vstack([a, np.broadcast_to(v, (days, a.shape[1])).astype(a.dtype)])
        return Book(
            accounts=self.accounts, ccy=self.ccy, acc_type=self.acc_type,
            dates=self.dates.append(pd.date_range(self.dates[-1] + pd.Timedelta(days=1), periods=days)),
            balance=pad(self.balance, np.nan), outflow=pad(self.outflow, np.nan), net_flow=pad(self.net_flow, np.nan),
            reliable=pad(self.reliable, True), avg_outflow=pad(self.avg_outflow, avg_last),
            floor=pad(self.floor, avg_last * cover), fx=self.fx,
        )

    # ---- primitivas de transferencia
    @property
    def n(self) -> int:
        return len(self.accounts)

    def kind(self, i: int, j: int) -> str:
        return "same_ccy" if self.ccy[i] == self.ccy[j] else "cross_ccy"

    def fee_rate(self, i: int, j: int) -> float:
        return cfg.FEE_RATE[self.kind(i, j)]

    def lag_plan(self, i: int, j: int) -> int:
        return cfg.LAG_PLAN[self.kind(i, j)]

    def fx_ratio(self, i: int, j: int, fx: np.ndarray | None = None) -> float:
        """Unidades de la moneda de ``j`` por unidad de la moneda de ``i``."""
        f = self.fx if fx is None else fx
        return f[j] / f[i]

    def date_at(self, k: int) -> pd.Timestamp:
        """Fecha de la posición ``k`` (permite posiciones posteriores al último dato; el calendario es diario)."""
        return self.dates[0] + pd.Timedelta(days=int(k))

    def to_usd(self, a: int, x):
        return x / self.fx[a]

    def min_size(self, t: int, a: int) -> float:
        """Monto mínimo de transferencia hacia ``a`` (moneda nativa de ``a``)."""
        return cfg.MIN_TRANSFER_DAYS_OF_OUTFLOW * self.avg_outflow[t + 1, a]


@dataclass
class Scheduled:
    """Transferencia ya solicitada. ``amount`` y ``fee`` en moneda del origen; ``credit`` en la del destino."""

    i: int
    j: int
    amount: float
    fee: float
    credit: float
    request_idx: int
    settle_idx: int


@dataclass
class Proposal:
    i: int
    j: int
    amount: float                      # moneda nativa del origen, sin fee
    reason: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class Context:
    t: int
    book: Book
    bal: np.ndarray                    # (A,) saldo simulado al cierre de t
    history: np.ndarray                # (t+1, A) saldos simulados hasta t
    pending: list[Scheduled]           # solicitadas con liquidación posterior a t
    forecast: OriginForecast | None
    rng: np.random.Generator

    def arrivals_cum(self, H: int) -> np.ndarray:
        """(H, A) acumulado de abonos ya comprometidos en t+1..t+H (moneda nativa del destino)."""
        arr = np.zeros((H, self.book.n))
        for s in self.pending:
            k = s.settle_idx - self.t
            if 1 <= k <= H:
                arr[k - 1 :, s.j] += s.credit
        return arr


# ------------------------------------------------------- política del squad (A)
class SquadRule:
    """Regla del squad tal cual: media 14d del NIVEL, umbral fijo (4 de 6 cuentas), donante = idxmax nominal.

    ``currency_aware=True`` (variante A2) corrige SOLO el defecto D6: monto convertido a la moneda del
    donante y donante = mayor saldo en USD equivalente. Aísla cuánto de la brecha es FX y cuánto es
    el resto (umbral fijo, forecast sin horizonte, ignorar tránsito y lag).
    """

    def __init__(self, currency_aware: bool = False):
        self.currency_aware = currency_aware
        self.name = "A2: squad + FX" if currency_aware else "A: regla del squad"

    def decide(self, ctx: Context) -> list[Proposal]:
        book, out = ctx.book, []
        if ctx.t + 1 < cfg.SQUAD_WINDOW:
            return out
        forecast = ctx.history[-cfg.SQUAD_WINDOW :].mean(axis=0)
        for acc, thr in cfg.SQUAD_THRESHOLD.items():
            a = book.accounts.index(acc)
            if forecast[a] >= thr:
                continue
            others = [d for d in range(book.n) if d != a]
            score = (ctx.bal / book.fx) if self.currency_aware else ctx.bal      # D6: saldo nominal sin FX
            donor = max(others, key=lambda d: score[d])
            need = thr - forecast[a]                                              # moneda del DESTINO
            amount = need * book.fx_ratio(a, donor) if self.currency_aware else need  # D6: mismo número, otra moneda
            out.append(Proposal(donor, a, float(amount), f"forecast 14d {forecast[a]:,.0f} < umbral {thr:,.0f}"))
        return out


# ------------------------------------------------------ política de cuantil (B)
class QuantilePolicy:
    """Buffer objetivo derivado de la distribución: el cuantil bajo del saldo proyectado.

    Para cada cuenta se proyecta el saldo de los próximos H días (saldo + flujo neto simulado +
    abonos en tránsito). Si el cuantil ``alpha`` de esa proyección cae bajo el piso en algún día
    alcanzable por una transferencia nueva (t+1+lag), se cubre el faltante con el/los donantes de
    mayor excedente, cuyo propio cuantil ``alpha`` debe seguir sobre SU piso tras donar.
    Preferencia de donante: reserva sobre operativa → misma moneda (0,04% y lag ≤1) → mayor excedente USD.
    El donante conserva además ``DONOR_MARGIN_DAYS`` de egresos sobre su piso (histéresis: no dona hoy lo que mañana necesitará).
    """

    def __init__(self, alpha: float = cfg.QUANTILE_POLICY_ALPHA, n_draws: int = 500, bias_sd: float = 0.0):
        self.alpha, self.n_draws, self.bias_sd = alpha, n_draws, bias_sd
        self.name = f"B: cuantil p{alpha * 100:g}"
        self.log: list[dict] = []                                         # decisiones de esperar (solo las usa la política D)

    def _projections(self, ctx: Context) -> tuple[np.ndarray, np.ndarray]:
        """(cuantil ``alpha``, mediana) del saldo proyectado a H días, ambos (H, A), calculados con los mismos sorteos."""
        H = cfg.DECISION_HORIZON
        paths = ctx.forecast.draw(self.n_draws, ctx.rng, self.bias_sd)
        proj = ctx.bal[None, None, :] + np.cumsum(paths, axis=1) + ctx.arrivals_cum(H)[None]
        return np.quantile(proj, self.alpha, axis=0), np.quantile(proj, 0.5, axis=0)

    def projected_quantile(self, ctx: Context) -> np.ndarray:
        return self._projections(ctx)[0]                                  # (H, A)

    def _ranked_donors(self, a: int, room: np.ndarray) -> list[int]:
        """Donantes de ``a`` con excedente: reserva sobre operativa, misma moneda primero, mayor excedente USD."""
        book = self._book
        donors = [d for d in range(book.n) if d != a and room[d] > 0]
        donors.sort(key=lambda d: (book.acc_type[d] != "reserve", book.kind(d, a) != "same_ccy", -room[d] / book.fx[d]))
        return donors

    def _wait_reason(self, ctx: Context, a: int, lag: int, q: np.ndarray, m: np.ndarray, floor: np.ndarray):
        """Gancho: ``None`` = transferir como siempre; ``(motivo, info)`` = esperar. La política B nunca espera."""
        return None

    def diagnostic(self, ctx: Context) -> pd.DataFrame:
        """Por cuenta: saldo, piso, mínimo del cuantil proyectado y primer día bajo el piso (para explicar la decisión)."""
        book, t = ctx.book, ctx.t
        q = self.projected_quantile(ctx)
        floor = book.floor[t + 1]
        rows = []
        for a in range(book.n):
            below = q[:, a] < floor[a]
            rows.append(
                {
                    "cuenta": book.accounts[a], "moneda": book.ccy[a], "tipo": book.acc_type[a],
                    "saldo_hoy": ctx.bal[a], "piso": floor[a], f"p{self.alpha * 100:g}_minimo_{cfg.DECISION_HORIZON}d": q[:, a].min(),
                    "primer_dia_bajo_piso": book.date_at(t + 1 + int(np.argmax(below))).date() if below.any() else None,
                    "excedente_donable": max(float((q[:, a] - floor[a]).min() - cfg.DONOR_MARGIN_DAYS * book.avg_outflow[t + 1, a]), 0.0),
                }
            )
        return pd.DataFrame(rows).set_index("cuenta")

    def decide(self, ctx: Context) -> list[Proposal]:
        book, t = ctx.book, ctx.t
        self._book = book
        q, m = self._projections(ctx)
        floor = book.floor[t + 1]
        margin = cfg.DONOR_MARGIN_DAYS * book.avg_outflow[t + 1]
        room = np.maximum((q - floor).min(axis=0) - margin, 0.0)          # excedente donable (moneda nativa)
        breach_day = [(np.argmax(q[:, a] < floor[a]) if (q[:, a] < floor[a]).any() else None) for a in range(book.n)]
        recipients = sorted([a for a in range(book.n) if breach_day[a] is not None], key=lambda a: (book.acc_type[a] != "operational", breach_day[a], -(floor[a] - q[:, a].min())))
        out: list[Proposal] = []
        for a in recipients:
            remaining = None
            donors = self._ranked_donors(a, room)
            if donors:
                wait = self._wait_reason(ctx, a, book.lag_plan(donors[0], a), q, m, floor)
                if wait:
                    self.log.append({"fecha": book.dates[t].date(), "cuenta": book.accounts[a], "moneda": book.ccy[a], "motivo": wait[0], **wait[1]})
                    continue
            for d in donors:
                kmin = 1 + book.lag_plan(d, a)                            # 1 = t+1 es el primer día de solicitud
                usable = q[kmin - 1 :, a]
                need = max(float(floor[a] - usable.min()), 0.0) if remaining is None else remaining
                if need <= 0:
                    break
                headroom = cfg.MIN_TRANSFER_DAYS_OF_OUTFLOW * book.avg_outflow[t + 1, a]
                credit = need + headroom
                ratio, fee = book.fx_ratio(d, a), book.fee_rate(d, a)
                amount = min(credit / ratio, room[d] / (1 + fee))
                if amount * ratio < book.min_size(t, a) * 0.999:
                    continue
                k_short = int(kmin - 1 + np.argmin(usable))
                reason = (
                    f"p{self.alpha * 100:g} del saldo de {book.accounts[a]} cae a {q[:, a].min():,.0f} {book.ccy[a]} "
                    f"(piso {floor[a]:,.0f}); primer faltante proyectado {book.date_at(t + 1 + int(breach_day[a])).date()}, "
                    f"cubre el {book.date_at(t + 1 + k_short).date()}; donante {book.accounts[d]} con excedente p{self.alpha * 100:g} "
                    f"{room[d]:,.0f} {book.ccy[d]}"
                )
                out.append(Proposal(d, a, float(amount), reason, {"cover_day": book.date_at(t + 1 + k_short)}))
                room[d] -= amount * (1 + fee)
                remaining = max(need - amount * ratio, 0.0)
                if remaining <= 1e-9:
                    break
        return out


# ------------------------------------------------- política D: cuantil + esperar si se recupera
class WaitAwarePolicy(QuantilePolicy):
    """Política B con una regla de espera: tocar el piso no obliga a mover dinero si el pronóstico central se recupera.

    Una transferencia solicitada hoy llega en ``lag`` días; solo puede cubrir faltantes de ese día en adelante. Para cada
    cuenta con faltante proyectado (cuantil ``alpha`` bajo el piso en un día cubrible) se pregunta:

    1. ¿Sobra tiempo? ``holgura`` = días entre lo que tarda en llegar el dinero y el primer día bajo el piso. Si hay al menos
       ``min_slack_days``, se puede esperar y solicitar más tarde sin llegar tarde.
    2. ¿El pronóstico central se recupera? La **mediana** proyectada se mantiene sobre el piso + ``recovery_margin_days`` de
       egreso en todos los días cubribles: el faltante solo existe en la cola de la distribución, no en el caso esperado.

    Si ambas se cumplen, **espera** y se re-evalúa mañana (reoptimización diaria); la decisión queda en ``log`` con el último
    día para solicitar. Si no, transfiere igual que B (mismos donantes, montos y margen). Es la versión de regla simple de lo
    que el MILP hace por costo: no mover dinero cuando esperar sale más barato.
    """

    def __init__(self, alpha: float = cfg.QUANTILE_POLICY_ALPHA, n_draws: int = 500, bias_sd: float = 0.0,
                 min_slack_days: int = cfg.WAIT_MIN_SLACK_DAYS, recovery_margin_days: float = cfg.WAIT_RECOVERY_MARGIN_DAYS):
        super().__init__(alpha, n_draws, bias_sd)
        self.min_slack_days, self.recovery_margin_days = min_slack_days, recovery_margin_days
        self.name = f"D: cuantil p{alpha * 100:g} + espera"

    def _assess(self, ctx: Context, a: int, lag: int, q: np.ndarray, m: np.ndarray, floor: np.ndarray) -> dict:
        """Estado de una cuenta: 'ok' (sin faltante cubrible), 'esperar' o 'transferir', con los números que lo explican."""
        book, t = ctx.book, ctx.t
        fixable_q, fixable_m = q[lag:, a], m[lag:, a]
        below = np.where(fixable_q < floor[a])[0]
        if not len(below):
            return {"estado": "ok", "holgura_dias": np.nan, "primer_dia_bajo_piso": None, "ultimo_dia_para_solicitar": None}
        slack = int(below[0])                                             # días que se puede esperar y aún llegar a tiempo
        breach_day = book.date_at(t + 1 + lag + slack)
        margin = self.recovery_margin_days * book.avg_outflow[t + 1, a]
        recovers = bool(fixable_m.min() >= floor[a] + margin)
        wait = slack >= self.min_slack_days and recovers
        return {"estado": "esperar" if wait else "transferir", "holgura_dias": slack, "primer_dia_bajo_piso": breach_day.date(),
                "ultimo_dia_para_solicitar": book.date_at(t + 1 + slack).date(), "mediana_se_recupera": recovers}

    def _wait_reason(self, ctx, a, lag, q, m, floor):
        r = self._assess(ctx, a, lag, q, m, floor)
        if r["estado"] != "esperar":
            return None
        book = ctx.book
        return (f"p{self.alpha * 100:g} de {book.accounts[a]} toca el piso {r['primer_dia_bajo_piso']} pero la mediana se mantiene sobre "
                f"piso + {self.recovery_margin_days:g} día(s) de egreso: se espera; último día para solicitar {r['ultimo_dia_para_solicitar']}",
                {"holgura_dias": r["holgura_dias"], "ultimo_dia_para_solicitar": r["ultimo_dia_para_solicitar"]})

    def opportunities(self, ctx: Context) -> pd.DataFrame:
        """Tabla por cuenta para Tesorería: qué hacer hoy, con cuánta holgura y hasta cuándo se puede esperar."""
        book, t = ctx.book, ctx.t
        self._book = book
        q, m = self._projections(ctx)
        floor = book.floor[t + 1]
        room = np.maximum((q - floor).min(axis=0) - cfg.DONOR_MARGIN_DAYS * book.avg_outflow[t + 1], 0.0)
        rows = []
        for a in range(book.n):
            donors = self._ranked_donors(a, room)
            lag = book.lag_plan(donors[0], a) if donors else cfg.LAG_PLAN["cross_ccy"]
            r = self._assess(ctx, a, lag, q, m, floor)
            rows.append({"cuenta": book.accounts[a], "moneda": book.ccy[a], "tipo": book.acc_type[a], "saldo_hoy": ctx.bal[a], "piso": floor[a],
                         f"p{self.alpha * 100:g}_minimo": q[:, a].min(), "mediana_minima": m[:, a].min(), "donante_preferido": book.accounts[donors[0]] if donors else None,
                         "lag_dias": lag, **{k: v for k, v in r.items() if k != "mediana_se_recupera"}})
        return pd.DataFrame(rows).set_index("cuenta")


# ------------------------------------------------------------- utilidades de plan
def plan_to_frame(props: list[Proposal], ctx: Context, executed_only: bool = True) -> pd.DataFrame:
    """Plan con origen, destino, monto, moneda, fecha de ejecución, liquidación esperada, fee y motivo."""
    book, rows = ctx.book, []
    for p in props:
        lag = book.lag_plan(p.i, p.j)
        request = p.info.get("request_idx", ctx.t + 1)
        day = lambda k: (book.dates[ctx.t] + pd.Timedelta(days=int(k))).date()
        rows.append(
            {
                "corte": book.dates[ctx.t].date(),
                "origen": book.accounts[p.i], "destino": book.accounts[p.j],
                "moneda_origen": book.ccy[p.i], "monto": p.amount,
                "fecha_ejecucion": day(request - ctx.t),
                "liquidacion_esperada": day(request - ctx.t + lag),
                "fee_estimado": p.amount * book.fee_rate(p.i, p.j),
                "moneda_destino": book.ccy[p.j],
                "monto_recibido": p.amount * book.fx_ratio(p.i, p.j),
                "fx_usada": f"{book.ccy[p.j]}/{book.ccy[p.i]} = {book.fx_ratio(p.i, p.j):,.6g}",
                "motivo": p.reason,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------- §4.4 sizing: colchón a H días, no solo hasta el piso
def target_buffer(ctx: Context, a: int, donor: int, alpha: float = cfg.QUANTILE_POLICY_ALPHA,
                   H: int = cfg.DECISION_HORIZON, n_draws: int = 2000) -> dict:
    """Cuánto transferir para que ``a`` no vuelva a tocar el piso en los próximos ``H`` días, no solo hoy.

    ``B̂_{a,t+L}`` se toma como el PEOR punto (mínimo) del cuantil ``alpha`` del saldo acumulado entre ``L`` y el
    final del horizonte del forecast — no solo el día exacto ``t+L`` — del bootstrap empírico (F3, ``ctx.forecast``),
    no de la media: es el sustituto directo de ``Threshold + Z_alpha·σ·√(L+1)`` sin asumir normalidad (nada le gana
    al empírico en 2.3), y del mismo criterio ya usado por ``QuantilePolicy``/``WaitAwarePolicy`` (una transferencia
    solicitada hoy solo puede cubrir faltantes desde ``t+1+L`` en adelante, así que hay que protegerse contra el
    peor día alcanzable, no solo el primero). ``Target_a(H) = piso_a + H·mu_burn_a`` añade encima el colchón de
    operación (``mu_burn``, el egreso neto medio del mismo forecast): ``monto = Target_a(H) − B̂_{a,t+L}``. A
    diferencia de ``QuantilePolicy.decide`` (que solo cubre el faltante más un margen mínimo,
    ``MIN_TRANSFER_DAYS_OF_OUTFLOW``), esto amortigua contra tener que volver a transferir mañana, al costo de
    dejar más saldo ocioso (ver ``backtest.metrics_by_currency``, columna ``saldo_ocioso_prom``).
    """
    book, t = ctx.book, ctx.t
    L = book.lag_plan(donor, a)
    floor = book.floor[t + 1, a]
    paths = ctx.forecast.draw(n_draws, ctx.rng)[:, :, a]                    # (n, H_forecast) trayectorias del flujo neto
    q_path = np.quantile(np.cumsum(paths, axis=1), alpha, axis=0)          # (H_forecast,) trayectoria del cuantil alpha
    q_worst = float(q_path[L:].min())                                       # peor día alcanzable por una transferencia pedida hoy
    b_hat = ctx.bal[a] + q_worst                                            # saldo "pesimista" (piso + colchón) en el peor día
    mu_burn = max(-float(ctx.forecast.mean[:, a].mean()), 0.0)              # egreso neto medio diario
    target = floor + H * mu_burn
    return {
        "lag_dias": L, "piso": floor, "b_hat_alpha": b_hat, "mu_burn_diario": mu_burn,
        "target_buffer_H": target, "monto_requerido": max(target - b_hat, 0.0),
    }
