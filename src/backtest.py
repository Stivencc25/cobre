"""F5 — Simulación día a día de políticas de rebalanceo con contrafactual y métricas por moneda.

El "contrafactual" es el saldo orgánico (sin transferencias): los datos muestran que el log de
transferencias no está reflejado en los saldos, así que el saldo observado ES la trayectoria sin
intervención. Cada política se aplica sobre esa misma trayectoria, con el mismo forecast, los mismos
números aleatorios (escenarios y lags) y el mismo criterio de faltante.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from .forecast import OriginForecast, fit_origin
from .policy import Book, Context, Proposal, Scheduled


class NoTransfers:
    """Contrafactual: no hacer nada."""

    name = "0: sin transferencias"

    def decide(self, ctx: Context) -> list[Proposal]:
        return []


class ForecastStore:
    """Forecast por origen, calculado una vez con datos hasta t (sin lookahead) y compartido por las políticas."""

    def __init__(self, flows: pd.DataFrame, model: str):
        self.flows, self.model, self._cache = flows, model, {}

    def __getitem__(self, t: int) -> OriginForecast:
        if t not in self._cache:
            self._cache[t] = fit_origin(self.flows, t, self.model)
        return self._cache[t]


@dataclass
class SimSettings:
    """Escenario de ejecución. Los valores por defecto son el caso base."""

    fx_shock: float = 0.0                # el FX REAL de ejecución = FX de config × (1 + shock) para COP y MXN
    lag_forced: int | None = None        # fuerza el lag realizado (p. ej. 3) sin avisar al planificador
    bias_sd: float = 0.0                 # se le pasa a las políticas que usan forecast
    label: str = "base"


@dataclass
class SimResult:
    policy: str
    settings: SimSettings
    book: Book
    t0: int
    balance: np.ndarray                  # (T, A) saldo simulado
    transfers: list[tuple[Scheduled, str]] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)


def lag_tables(T: int) -> dict[str, np.ndarray]:
    """Lag realizado por día de solicitud y tipo de par, muestreado de la distribución observada (misma semilla para todas las políticas)."""
    rng = np.random.default_rng(cfg.BACKTEST_LAG_SEED)
    out = {}
    for kind, freq in cfg.LAG_OBSERVED.items():
        lags, w = np.array(list(freq.keys())), np.array(list(freq.values()), float)
        out[kind] = rng.choice(lags, size=T + cfg.LAG_STRESS_DAYS + 5, p=w / w.sum())
    return out


def simulate(book: Book, policy, store: ForecastStore | None, settings: SimSettings = SimSettings(), t0: int | None = None) -> SimResult:
    """Corre la política con reoptimización diaria desde ``t0`` hasta el penúltimo día."""
    T, A = book.balance.shape
    t0 = cfg.MIN_TRAIN_DAYS - 1 if t0 is None else t0
    fx_real = book.fx * np.array([1.0 if c == "USD" else 1.0 + settings.fx_shock for c in book.ccy])
    lags = lag_tables(T)
    delta = np.zeros((T + 1, A))
    sim_bal = book.balance.copy()
    pending: list[Scheduled] = []
    transfers: list[tuple[Scheduled, str]] = []
    cum = np.zeros(A)
    if hasattr(policy, "bias_sd"):
        policy.bias_sd = settings.bias_sd
    for t in range(T):
        cum = cum + delta[t]
        sim_bal[t] = book.balance[t] + cum
        if t < t0 or t >= T - 1:
            continue
        pending = [s for s in pending if s.settle_idx > t]
        ctx = Context(t, book, sim_bal[t].copy(), sim_bal[: t + 1].copy(), list(pending),
                      store[t] if store is not None else None, np.random.default_rng([cfg.BOOTSTRAP_SEED, t]))
        remaining = sim_bal[t].copy()
        for p in policy.decide(ctx):
            fee_rate = book.fee_rate(p.i, p.j)
            amount = min(p.amount, max(remaining[p.i], 0.0) / (1 + fee_rate))    # el banco no gira fondos que no hay
            if amount <= 0:
                continue
            kind = book.kind(p.i, p.j)
            lag = settings.lag_forced if settings.lag_forced is not None else int(lags[kind][t + 1])
            sched = Scheduled(p.i, p.j, amount, amount * fee_rate, amount * fx_real[p.j] / fx_real[p.i], t + 1, t + 1 + lag)
            remaining[p.i] -= amount * (1 + fee_rate)
            delta[t + 1, p.i] -= sched.amount + sched.fee
            if sched.settle_idx < T:
                delta[sched.settle_idx, p.j] += sched.credit
            pending.append(sched)
            transfers.append((sched, p.reason))
    return SimResult(policy.name, settings, book, t0, sim_bal, transfers, getattr(policy, "log", []))


# --------------------------------------------------------------------- métricas
def deficits(res: SimResult) -> np.ndarray:
    """(T, A) déficit vs piso; NaN fuera de la ventana de evaluación (t0+1 ..)."""
    d = np.maximum(res.book.floor - res.balance, 0.0)
    d[: res.t0 + 1] = np.nan
    return d


def metrics_by_currency(res: SimResult) -> pd.DataFrame:
    """Métricas obligatorias por moneda (nunca agregadas entre monedas)."""
    book, d = res.book, deficits(res)
    idle = np.maximum(res.balance - book.floor * (1 + cfg.IDLE_MARGIN), 0.0)
    idle[: res.t0 + 1] = np.nan
    rows = []
    for ccy in sorted(set(book.ccy)):
        cols = [a for a in range(book.n) if book.ccy[a] == ccy]
        tr = [s for s, _ in res.transfers if book.ccy[s.i] == ccy]
        rows.append(
            {
                "politica": res.policy, "moneda": ccy,
                "dias_cuenta_en_faltante": int(np.nansum(d[:, cols] > 0)),
                "deficit_total": float(np.nansum(d[:, cols])),
                "n_transferencias": len(tr),
                "fees_totales": float(sum(s.fee for s in tr)),
                "saldo_ocioso_prom": float(np.nanmean(idle[:, cols])),
            }
        )
    return pd.DataFrame(rows)


def totals_usd(res: SimResult) -> dict:
    """Totales en USD equivalente a FX de config (solo para la frontera; las métricas oficiales son por moneda).

    ``penalizacion_usd`` es la misma fórmula que usa el objetivo del MILP (``cfg.DEFICIT_PENALTY_PER_DAY``
    ponderado por ``cfg.PENALTY_WEIGHT`` del tipo de cuenta) aplicada al déficit REALIZADO en el backtest:
    lo que habría costado, en la misma unidad que los fees, no cubrir cada faltante en vez de transferir.
    """
    book, d = res.book, deficits(res)
    usd = 1.0 / book.fx
    w = np.array([cfg.PENALTY_WEIGHT[t] for t in book.acc_type])
    fees = sum(s.fee * usd[s.i] for s, _ in res.transfers)
    n = len(res.transfers)
    return {
        "politica": res.policy, "escenario": res.settings.label,
        "n_transferencias": n, "fees_usd": fees, "costo_fijo_usd": n * cfg.FIXED_COST_USD,
        "costo_total_usd": fees + n * cfg.FIXED_COST_USD,
        "deficit_usd_dias": float(np.nansum(d * usd)),
        "dias_cuenta_en_faltante": int(np.nansum(d > 0)),
        "penalizacion_usd": float(np.nansum(d * usd[None, :] * w[None, :])) * cfg.DEFICIT_PENALTY_PER_DAY,
        "volumen_usd": sum(s.amount * usd[s.i] for s, _ in res.transfers),
    }


def run_policies(book: Book, store: ForecastStore, policies: list, settings: SimSettings = SimSettings()) -> dict[str, SimResult]:
    return {p.name: simulate(book, p, store, settings) for p in policies}


def compare(results: dict[str, SimResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tabla por moneda, tabla de totales USD)."""
    by_ccy = pd.concat([metrics_by_currency(r) for r in results.values()], ignore_index=True)
    tot = pd.DataFrame([totals_usd(r) for r in results.values()])
    return by_ccy, tot


# --------------------------------------------------- ejecución en paralelo y barridos
def _run_job(job):
    book, policy, store, settings = job
    res = simulate(book, policy, store, settings)
    res.log = list(getattr(policy, "log", []))
    res.plan = getattr(policy, "last_plan", [])
    return res


def simulate_many(book: Book, store: ForecastStore, jobs: list[tuple], workers: int | None = None) -> list[SimResult]:
    """Corre ``[(política, settings), ...]`` en procesos (spawn) manteniendo el orden. Cada job es independiente."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    payload = [(book, p, store, s) for p, s in jobs]
    if workers == 1 or len(payload) == 1:
        return [_run_job(j) for j in payload]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_run_job, payload))


def frontier(book: Book, store: ForecastStore, lp_grid=cfg.LP_PENALTY_GRID, alpha_grid=cfg.QUANTILE_ALPHA_GRID,
             settings: SimSettings = SimSettings(), workers: int | None = None) -> tuple[pd.DataFrame, dict[str, SimResult]]:
    """Todas las políticas sobre los mismos datos: contrafactual, squad (A, A2), cuantil (B) y MILP (C) con sus barridos."""
    from .optimize import LPPolicy
    from .policy import QuantilePolicy, SquadRule

    policies = [NoTransfers(), SquadRule(), SquadRule(currency_aware=True)]
    policies += [QuantilePolicy(a) for a in alpha_grid] + [LPPolicy(l) for l in lp_grid]
    results = simulate_many(book, store, [(p, settings) for p in policies], workers)
    fam = lambda n: n[0]
    rows = []
    for r in results:
        row = totals_usd(r)
        row["familia"] = fam(r.policy)
        rows.append(row)
    return pd.DataFrame(rows), {r.policy: r for r in results}


def sensitivity(panel: pd.DataFrame, store: ForecastStore, workers: int | None = None) -> pd.DataFrame:
    """FX ±10%, lag forzado a 3, forecast sesgado y piso de 30 días, para squad (A), cuantil p5 (B) y MILP (C)."""
    from .optimize import LPPolicy
    from .policy import QuantilePolicy, SquadRule

    base_book = Book.from_panel(panel)
    hard_book = Book.from_panel(panel, cover_days={"operational": 30.0, "reserve": 30.0})
    cases = [
        ("base", base_book, SimSettings(label="base")),
        (f"FX +{cfg.SENSITIVITY_FX_SHOCK:.0%}", base_book, SimSettings(fx_shock=cfg.SENSITIVITY_FX_SHOCK, label=f"FX +{cfg.SENSITIVITY_FX_SHOCK:.0%}")),
        (f"FX -{cfg.SENSITIVITY_FX_SHOCK:.0%}", base_book, SimSettings(fx_shock=-cfg.SENSITIVITY_FX_SHOCK, label=f"FX -{cfg.SENSITIVITY_FX_SHOCK:.0%}")),
        (f"lag forzado {cfg.LAG_STRESS_DAYS}d", base_book, SimSettings(lag_forced=cfg.LAG_STRESS_DAYS, label=f"lag forzado {cfg.LAG_STRESS_DAYS}d")),
        (f"forecast sesgado +{cfg.SENSITIVITY_BIAS_SD}σ", base_book, SimSettings(bias_sd=cfg.SENSITIVITY_BIAS_SD, label=f"forecast sesgado +{cfg.SENSITIVITY_BIAS_SD}σ")),
        ("piso 30 días", hard_book, SimSettings(label="piso 30 días")),
    ]
    jobs, meta = [], []
    for label, bk, st in cases:
        for pol in (NoTransfers(), SquadRule(), QuantilePolicy(), LPPolicy()):
            jobs.append((pol, st))
            meta.append(bk)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    payload = [(bk, p, store, s) for bk, (p, s) in zip(meta, jobs)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_run_job, payload))
    return pd.DataFrame([totals_usd(r) for r in results])


# ------------------------------------------------------------------- utilidades
def context_at(book: Book, store: ForecastStore, t: int, balance: np.ndarray | None = None,
               pending: list[Scheduled] | None = None) -> Context:
    """Contexto de decisión al cierre de ``t`` (por defecto, con el saldo orgánico y sin transferencias en tránsito)."""
    bal = book.balance[t].copy() if balance is None else balance
    return Context(t, book, bal, book.balance[: t + 1].copy(), pending or [], store[t], np.random.default_rng([cfg.BOOTSTRAP_SEED, t]))


def realized_deficit_usd(res: SimResult, account: str, date, days: int = cfg.DECISION_HORIZON) -> float:
    """Déficit realizado (USD·día equivalente) de ``account`` en los ``days`` días posteriores a ``date``."""
    book = res.book
    a, t = book.accounts.index(account), int(np.where(book.dates == pd.Timestamp(date))[0][0])
    sl = slice(t + 1, min(t + 1 + days, len(book.dates)))
    return float(np.maximum(book.floor[sl, a] - res.balance[sl, a], 0.0).sum() / book.fx[a])
