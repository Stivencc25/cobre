"""F4 (capa 2) — Optimización MILP a horizonte rodante, por aproximación de escenarios (SAA).

Problema (todo en USD equivalente a FX de config; se reconvierte a moneda nativa al emitir el plan):

    min   Σ fee_ij·x_ij,d  +  F·Σ z_ij,d  +  ε·Σ d·x_ij,d  +  λ · Σ_a w_a · (1/S) Σ_k Σ_s u_a,k,s
    s.a.  u_a,k,s ≥ piso_a − saldo_a,k,s                           (déficit blando por escenario;
                                                                        en la matriz: u − Σsalidas·(1+fee) + Σentradas ≥ piso − saldo_sin_transferir)
          saldo_a,k,s = saldo_a,0 + Σ_{m≤k} flujo_a,m,s + abonos_pendientes_a,k
                        − Σ_{j,d≤k}(1+fee_aj)·x_aj,d + Σ_{i,d: d+lag_ia≤k} x_ia,d
          Σ_{j,d}(1+fee_aj)·x_aj,d ≤ max(0, saldo_a,0 − piso_a)    (el donante no cae bajo su piso hoy)
          mín_j·z ≤ x ≤ M·z ,  z ∈ {0,1}                          (costo fijo y monto mínimo)

Los fondos llegan en d+lag (lag de planificación por tipo de par), no en d. Solo se ejecuta d=1;
el resto del plan se recalcula mañana (horizonte rodante). ε desempata a favor de transferir antes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from . import config as cfg
from .policy import Context, Proposal


class LPPolicy:
    def __init__(self, penalty: float = cfg.DEFICIT_PENALTY_PER_DAY, n_scen: int = cfg.N_SCENARIOS_LP,
                 bias_sd: float = 0.0, fixed_cost: float = cfg.FIXED_COST_USD):
        self.penalty, self.n_scen, self.bias_sd, self.fixed_cost = penalty, n_scen, bias_sd, fixed_cost
        self.name = f"C: MILP λ={penalty:g}"
        self.log: list[dict] = []           # decisiones "no transferir pese a faltante proyectado"
        self.last_plan: list[Proposal] = []  # plan completo (todas las fechas de solicitud) de la última corrida

    # ------------------------------------------------------------------
    def decide(self, ctx: Context) -> list[Proposal]:
        book, t = ctx.book, ctx.t
        H, A = cfg.DECISION_HORIZON, book.n
        usd = 1.0 / book.fx
        paths = ctx.forecast.draw(self.n_scen, ctx.rng, self.bias_sd) * usd     # (S,H,A) USD
        S = paths.shape[0]
        bal0 = ctx.bal * usd
        floor = book.floor[t + 1] * usd
        arr = ctx.arrivals_cum(H) * usd                                           # (H,A)
        base = bal0[None, None, :] + np.cumsum(paths, axis=1) + arr[None]         # saldo sin transferencias
        gap = floor[None, None, :] - base                                         # >0 => déficit sin transferir
        breached = [a for a in range(A) if (gap[:, :, a] > 0).any()]
        self.last_plan = []
        if not breached:
            return []
        # Prefiltro exacto: transferir hacia ``a`` no puede evitar más que su penalización esperada; si esa
        # cota no supera el costo fijo de una operación, ninguna transferencia hacia ``a`` puede ser rentable.
        exp_pen = {a: self.penalty * cfg.PENALTY_WEIGHT[book.acc_type[a]] * float(np.maximum(gap[:, :, a], 0).sum(axis=1).mean())
                   for a in breached}
        recipients = [a for a in breached if exp_pen[a] > self.fixed_cost]
        excess = np.maximum(bal0 - floor, 0.0)
        donors = [i for i in range(A) if excess[i] > 0]
        pairs = [(i, j) for i in donors for j in recipients if i != j]
        if not pairs:
            self._log_skip(ctx, gap, breached, "penalización esperada ≤ costo fijo de operar" if not recipients else "sin donante con excedente")
            return []

        var = []                                     # (i, j, d): origen, destino, día de solicitud (1 = t+1)
        for (i, j) in pairs:
            L = book.lag_plan(i, j)
            for d in range(1, min(cfg.LP_MAX_REQUEST_DAYS, H - L) + 1):
                var.append((i, j, d))
        nx = len(var)
        involved = sorted({a for i, j, _ in var for a in (i, j)})
        na = len(involved)
        pos = {a: n for n, a in enumerate(involved)}
        nu = na * H * S
        n_total = 2 * nx + nu                        # [x | z | u]

        # ---- objetivo
        c = np.zeros(n_total)
        for v, (i, j, d) in enumerate(var):
            c[v] = book.fee_rate(i, j) + cfg.TIEBREAK_EARLY * d
            c[nx + v] = self.fixed_cost
        w = np.array([cfg.PENALTY_WEIGHT[book.acc_type[a]] for a in involved])
        c[2 * nx :] = np.repeat(self.penalty * w / S, H * S)

        # ---- restricciones: déficit por (cuenta, día, escenario), vectorizado
        cx = np.zeros((na, H, nx))
        for v, (i, j, d) in enumerate(var):
            L = book.lag_plan(i, j)
            cx[pos[i], d - 1 :, v] -= 1.0 + book.fee_rate(i, j)
            cx[pos[j], d - 1 + L :, v] += 1.0
        pa, kk, vv = np.nonzero(cx)
        base_rows = (pa * H + kk) * S
        rows = (base_rows[:, None] + np.arange(S)[None, :]).ravel()
        cols = np.repeat(vv, S)
        vals = np.repeat(cx[pa, kk, vv], S)
        rows_u = np.arange(nu)
        rows = np.concatenate([rows, rows_u])
        cols = np.concatenate([cols, 2 * nx + rows_u])
        vals = np.concatenate([vals, np.ones(nu)])
        lo_list = [gap[:, :, involved].transpose(2, 1, 0).reshape(-1)]
        hi_list = [np.full(nu, np.inf)]

        r = nu
        er, ec, ev, elo, ehi = [], [], [], [], []
        def add(row_entries, lo_, hi_):
            nonlocal r
            for col, val in row_entries:
                er.append(r); ec.append(col); ev.append(val)
            elo.append(lo_); ehi.append(hi_)
            r += 1

        for v, (i, j, d) in enumerate(var):
            add([(v, 1.0), (nx + v, -excess[i])], -np.inf, 0.0)                          # x ≤ M z
            add([(v, 1.0), (nx + v, -book.min_size(t, j) * usd[j])], 0.0, np.inf)        # x ≥ mín z
        for i in donors:                                                                  # donante no cae bajo su piso hoy
            add([(v, 1.0 + book.fee_rate(ii, j)) for v, (ii, j, d) in enumerate(var) if ii == i], -np.inf, excess[i])
        rows = np.concatenate([rows, er]); cols = np.concatenate([cols, ec]); vals = np.concatenate([vals, ev])
        lo = np.concatenate(lo_list + [elo]); hi = np.concatenate(hi_list + [ehi])
        Aeq = coo_matrix((vals, (rows, cols)), shape=(r, n_total)).tocsr()

        integrality = np.zeros(n_total)
        integrality[nx : 2 * nx] = 1
        ub = np.full(n_total, np.inf)
        ub[nx : 2 * nx] = 1.0
        res = milp(c, constraints=LinearConstraint(Aeq, lo, hi), integrality=integrality, bounds=Bounds(np.zeros(n_total), ub),
                   options={"time_limit": cfg.LP_TIME_LIMIT_S, "mip_rel_gap": cfg.LP_MIP_GAP})
        if res.x is None:
            raise RuntimeError(f"MILP sin solución en t={t}: {res.message}")

        plan: list[Proposal] = []
        x = res.x[:nx]
        for v, (i, j, d) in enumerate(var):
            if x[v] > 1e-6 * max(1.0, excess[i]) and res.x[nx + v] > 0.5:
                amount_native = float(x[v] * book.fx[i])
                reason = (f"MILP: reduce déficit esperado de {book.accounts[j]} "
                          f"(≈{float(np.maximum(gap[:, :, j], 0).sum(axis=1).mean()):,.0f} USD·día sin transferir, "
                          f"penalización esperada ≈ USD {exp_pen[j]:,.0f} > costo de operar); {book.accounts[i]} conserva su piso hoy")
                plan.append(Proposal(i, j, amount_native, reason, {"request_idx": t + d, "d": d}))
        self.last_plan = plan
        today = [p for p in plan if p.info["d"] == 1]
        if not plan:
            self._log_skip(ctx, gap, breached, "el MILP no transfiere: costo > penalización evitada")
        return today

    # ------------------------------------------------------------------
    def _log_skip(self, ctx: Context, gap: np.ndarray, recipients: list[int], why: str) -> None:
        """Deja constancia de 'faltante proyectado pero no transferir' con la comparación de costos."""
        book, t = ctx.book, ctx.t
        for a in recipients:
            w = cfg.PENALTY_WEIGHT[book.acc_type[a]]
            exp_def_days = float(np.mean(np.maximum(gap[:, :, a], 0).sum(axis=1)))
            penalty = self.penalty * w * exp_def_days
            need = float(np.mean(np.maximum(gap[:, :, a], 0).max(axis=1)))
            best_fee = min((book.fee_rate(i, a) for i in range(book.n) if i != a), default=cfg.FEE_RATE["cross_ccy"])
            cover_cost = self.fixed_cost + best_fee * need
            self.log.append(
                {"fecha": book.dates[t].date(), "cuenta": book.accounts[a], "moneda": book.ccy[a],
                 "prob_faltante_algun_dia": float((gap[:, :, a] > 0).any(axis=1).mean()),
                 "deficit_esperado_usd_dias": exp_def_days, "penalizacion_esperada_usd": penalty,
                 "costo_minimo_de_cubrir_usd": cover_cost, "motivo": why}
            )
