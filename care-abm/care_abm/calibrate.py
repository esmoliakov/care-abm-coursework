"""Profilių kalibravimas (3.1 sk.): randa savivaldybės biudžetą vienam senjorui, su kuriuo
S1 (low) ir S2 (high) R1 atitinka SADM taikinius (41,9 ir 312,7 / 1000).

Pastaba: tekste kalibruojami svertai, tačiau 7,5 karto skirtingų rodiklių tik trimis svertais
gauti neįmanoma – profilius realiai skiria biudžetas (savivaldybės būsenos kintamasis).
Svertai fiksuoti numatytosiomis reikšmėmis (DEFAULT_THETA).

Naudojimas:  python -m care_abm.calibrate [--seeds 40] [--tol 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import DEFAULT_THETA, TARGET_R1, SimParams
from .model import run_one
from .rng import sim_seed

OUT = Path(__file__).resolve().parent.parent / "configs" / "profiles.json"


def mean_r1(budget: float, seeds, base: SimParams) -> tuple[float, float]:
    p = replace(base, budget_per_senior=budget)
    r = np.array([run_one(p, DEFAULT_THETA, s)["R1"] for s in seeds])
    return float(r.mean()), float(r.std(ddof=1))


def calibrate_profile(target: float, seeds, base: SimParams, tol: float = 0.5,
                      lo: float = 0.005, hi: float = 10.0, max_iter: int = 60) -> dict:
    ceiling, _ = mean_r1(hi, seeds, base)
    if ceiling < target:
        raise RuntimeError(f"Taikinys {target} nepasiekiamas (lubos {ceiling:.1f}). "
                           f"Padidinkite poreikio dalį (h_beta_*) arba p_apply_with_family.")
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)                       # logaritminis dalijimas
        r, sd = mean_r1(mid, seeds, base)
        if abs(r - target) <= tol:
            return {"budget": mid, "R1_mean": r, "R1_sd": sd, "ceiling": ceiling}
        lo, hi = (mid, hi) if r < target else (lo, mid)
    r, sd = mean_r1(mid, seeds, base)
    return {"budget": mid, "R1_mean": r, "R1_sd": sd, "ceiling": ceiling}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--base-seed", type=int, default=20260)
    a = ap.parse_args()
    seeds = [sim_seed(a.base_seed, 5000, j) for j in range(a.seeds)]   # kalibravimo sėklos (atskirtos nuo eksperimentų)
    base = SimParams()
    res = {}
    for prof, target in TARGET_R1.items():
        res[prof] = calibrate_profile(target, seeds, base, a.tol)
        print(f"{prof:5s} taikinys={target:6.1f}  gauta R1={res[prof]['R1_mean']:6.1f} "
              f"(sd={res[prof]['R1_sd']:.1f}, n={a.seeds})  biudžetas/senjorui/mėn.={res[prof]['budget']:.4f}  "
              f"lubos={res[prof]['ceiling']:.1f}")
    out = {"budget_per_senior": {k: v["budget"] for k, v in res.items()},
           "details": res, "theta_default": DEFAULT_THETA, "n_seeds": a.seeds, "tol": a.tol,
           "note": "Kalibravimo rezultatai 3.1 sk. lentelei (tikslumas = |R1 - taikinys|, sd = atsitiktinis svyravimas)."}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Įrašyta:", OUT)


if __name__ == "__main__":
    main()