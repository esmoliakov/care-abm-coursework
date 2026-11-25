"""Tyrėjai be DKM: fiksuoti parametrai, atsitiktinė paieška, taisyklėmis grįstas tyrėjas."""
from __future__ import annotations

import numpy as np

from ..config import BOUNDS, NORM_MIN, NORM_TARGET, POLICY_KEYS
from .base import Proposal, Researcher


class FixedResearcher(Researcher):
    name = "fixed"
    iterative = False

    def propose(self, k, n_iter, theta, ind, history, ctx):
        return Proposal(dict(theta))


class RandomSearchResearcher(Researcher):
    """S3: kiekvienoje iteracijoje parametrai atsitiktinai (tolygiai) leistinose ribose."""
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def propose(self, k, n_iter, theta, ind, history, ctx):
        new = {key: float(self.rng.uniform(*BOUNDS[key])) for key in POLICY_KEYS}
        new["l_max"] = float(round(new["l_max"]))
        return Proposal(new)


def _clip(key, v):
    lo, hi = BOUNDS[key]
    return float(min(max(v, lo), hi))


class RuleBasedResearcher(Researcher):
    """S4 (full=False): jei R1 < 61, didinti alpha_amb (kaip 2.3 sk. lentelėje).
    S4b (full=True): sąžiningesnis palyginimas – taisyklė gali keisti visus tris svertus."""

    def __init__(self, full: bool = False):
        self.full = full
        self.name = "rule_full" if full else "rule"

    def propose(self, k, n_iter, theta, ind, history, ctx):
        th = dict(theta)
        r1 = ind["R1"]
        if not self.full:
            if r1 < NORM_MIN:
                th["alpha_amb"] = _clip("alpha_amb", th["alpha_amb"] + 0.10)
            return Proposal(th)
        if r1 < NORM_MIN:
            if th["alpha_amb"] < BOUNDS["alpha_amb"][1] - 1e-9:
                th["alpha_amb"] = _clip("alpha_amb", th["alpha_amb"] + 0.10)
            else:
                th["l_max"] = _clip("l_max", th["l_max"] + 2)
        elif ind["R5_share"] > 10.0 and th["respite"] < BOUNDS["respite"][1]:
            th["respite"] = _clip("respite", th["respite"] + 0.05)       # perdegimo prevencija
        elif ind["R4_need"] > 20.0:
            th["alpha_amb"] = _clip("alpha_amb", th["alpha_amb"] - 0.05)  # daugiau stacionaro vietų
        elif r1 < NORM_TARGET:
            th["l_max"] = _clip("l_max", th["l_max"] + 1)
        return Proposal(th)
