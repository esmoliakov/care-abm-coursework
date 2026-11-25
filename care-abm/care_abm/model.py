"""CareModel: mėnesio žingsnių seka, agentų sukūrimas, rodiklių rinkimas."""
from __future__ import annotations

import numpy as np

from .agents import Families, Municipality, Seniors, Workers
from .indicators import aggregate
from .processes import (allocate, apply_family_shock, demography, informal_care,
                        init_population, need_change)
from .rng import make_streams

SERIES_KEYS = ("alive", "need", "recip", "home", "inst", "aspn", "unmet", "queue", "wait_mean",
               "B_mean", "burned", "fam_n", "deaths", "spend", "budget", "respite", "maxload_social")


class CareModel:
    def __init__(self, params, theta: dict, seed: int):
        self.p = params
        self.theta = dict(theta)
        self.rng = make_streams(seed)
        self.month = 0
        self.deaths_this_month = 0
        lam = params.n_seniors * params.entrant_rate_annual / 12.0
        cap = params.n_seniors + int(params.months * lam * 2.5) + 200
        self.S, self.F = Seniors(cap), Families(cap)
        self.W = Workers(params.n_workers, params.n_social_workers)
        self.M = Municipality(params, theta)
        self.series = {k: [] for k in SERIES_KEYS}
        init_population(self)

    def step(self):
        self.month += 1
        if self.month == self.p.shock_month:
            apply_family_shock(self)
        demography(self)         # 1
        need_change(self)        # 2
        informal_care(self)      # 3
        allocate(self)           # 4 + 5 (priskyrimas, eilė)
        self._collect()          # 6

    def run(self):
        for _ in range(self.p.months):
            self.step()
        return self.series

    def _collect(self):
        S, F, p = self.S, self.F, self.p
        alive = S.alive
        n_alive = int(alive.sum())
        fam_ok = F.ok(alive, p.burnout_threshold)
        need_mask = alive & (S.need >= 1)
        f = S.formal
        met = (~need_mask) | ((S.need == 1) & ((f > 0) | fam_ok | S.aspn)) | \
              ((S.need == 2) & ((f == 2) | ((f == 1) & fam_ok)))
        queue = S.applicant & (f == 0) & need_mask
        fam_alive = F.exists & alive
        home, inst = int((f == 1).sum()), int((f == 2).sum())
        n_resp = int(F.resp_prev.sum())
        budget = self.M.budget(self.month)
        rec = dict(
            alive=n_alive,
            need=int(need_mask.sum()),
            recip=home + inst,
            home=home,
            inst=inst,
            aspn=int(S.aspn.sum()),
            unmet=int((need_mask & ~met).sum()),
            queue=int(queue.sum()),
            wait_mean=float(S.wait[queue].mean()) if queue.any() else 0.0,
            B_mean=float(F.B[fam_alive].mean()) if fam_alive.any() else 0.0,
            burned=float((F.B[fam_alive] >= p.burnout_threshold).mean()) if fam_alive.any() else 0.0,
            fam_n=int(fam_alive.sum()),
            deaths=self.deaths_this_month,
            spend=home * p.cost_home + inst * p.cost_inst + n_resp * p.cost_respite,
            budget=budget,
            respite=n_resp,
            maxload_social=self.W.max_load(0),
        )
        for k, v in rec.items():
            self.series[k].append(v)


def run_one(params, theta: dict, seed: int, return_series: bool = False):
    m = CareModel(params, theta, seed)
    series = m.run()
    ind = aggregate(series, params)
    return (ind, series) if return_series else ind


def evaluate(params, theta: dict, seeds) -> dict:
    """Vidutiniai rodikliai per kelias simuliacijas (vidinių pakartojimų m)."""
    inds = [run_one(params, theta, int(s)) for s in seeds]
    return {k: float(np.mean([i[k] for i in inds])) for k in inds[0]}
