"""Agentų tipai (masyvų struktūra: vienas masyvo elementas = vienas agentas).

Senolis, Šeimos narys (susietas su senolio vieta), Slaugytojas/darbuotojas, Savivaldybės atstovas.
"""
from __future__ import annotations

import math

import numpy as np

from .config import BOUNDS


class Seniors:
    """formal: 0 – nėra, 1 – pagalba į namus (socialinė šaka), 2 – stacionari globa."""

    def __init__(self, cap: int):
        self.cap = cap
        self.n_used = 0
        self.alive = np.zeros(cap, bool)
        self.age = np.zeros(cap)
        self.H = np.zeros(cap)
        self.need = np.zeros(cap, np.int8)        # 0 nereikia, 1 ambulatorinė, 2 stacionari
        self.formal = np.zeros(cap, np.int8)
        self.aspn = np.zeros(cap, bool)           # sveikatos šaka (ASPN)
        self.wait = np.zeros(cap, np.int32)
        self.admit = np.zeros(cap, np.int32)
        self.applicant = np.zeros(cap, bool)
        self.u_apply = np.zeros(cap)              # asmeninis polinkis teikti prašymą

    def add(self, age, H, need):
        n = len(age)
        if self.n_used + n > self.cap:
            raise RuntimeError("Senolių masyvo talpa per maža (padidinkite cap).")
        idx = np.arange(self.n_used, self.n_used + n)
        self.n_used += n
        self.alive[idx] = True
        self.age[idx] = age
        self.H[idx] = H
        self.need[idx] = need
        return idx


class Families:
    """Šeimos narys, susietas su senolio vieta (exists=False: emigravo / nebėra)."""

    def __init__(self, cap: int):
        self.exists = np.zeros(cap, bool)
        self.B = np.zeros(cap)                    # perdegimo lygis B_f in [0,1]
        self.resp_prev = np.zeros(cap, bool)      # gavo laikino atokvėpio paslaugą (paskutinį mėnesį)

    def ok(self, alive, thr):
        """Šeima pajėgi teikti neformalią globą."""
        return self.exists & alive & (self.B < thr)


class Workers:
    """Darbuotojai: branch 0 – socialinė šaka, 1 – sveikatos šaka (ASPN)."""

    def __init__(self, n_total: int, n_social: int):
        self.branch = np.r_[np.zeros(n_social, int), np.ones(n_total - n_social, int)]
        self.load = np.zeros(n_total, int)

    def assign(self, n_social_clients: int, n_health_clients: int):
        for br, n in ((0, n_social_clients), (1, n_health_clients)):
            idx = np.flatnonzero(self.branch == br)
            base, rem = divmod(int(n), len(idx))
            self.load[idx] = base
            self.load[idx[:rem]] += 1

    def max_load(self, branch: int) -> int:
        m = self.branch == branch
        return int(self.load[m].max()) if m.any() else 0


class Municipality:
    """Savivaldybės atstovas: biudžetas + politikos svertai theta (keičia tyrėjas)."""

    def __init__(self, params, theta: dict):
        self.p = params
        self.theta = dict(theta)

    def budget(self, t: int) -> float:
        b = self.p.budget_per_senior * self.p.n_seniors
        if self.p.shock_budget > 0 and t >= self.p.shock_month:
            b *= (1.0 - self.p.shock_budget)
        return b

    def capacities(self, t: int):
        p, th = self.p, self.theta
        a = min(max(th["alpha_amb"], BOUNDS["alpha_amb"][0]), BOUNDS["alpha_amb"][1])
        L = min(max(th["l_max"], BOUNDS["l_max"][0]), BOUNDS["l_max"][1])
        r = min(max(th["respite"], BOUNDS["respite"][0]), BOUNDS["respite"][1])
        B = self.budget(t)
        care = (1.0 - r) * B
        cap_home = min(math.floor(a * care / p.cost_home), p.n_social_workers * int(round(L)))
        cap_inst = math.floor((1.0 - a) * care / p.cost_inst)
        cap_resp = math.floor(r * B / p.cost_respite)
        return int(cap_home), int(cap_inst), int(cap_resp)
