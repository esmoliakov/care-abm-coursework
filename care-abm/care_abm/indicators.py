"""Rodiklių R1–R11 skaičiavimas (3 skyriaus indicators.py) ir tyrėjo kokybės funkcija.

Rodikliai skaičiuojami kaip paskutinių `window` (=12) mėnesių vidurkiai.
R1 apibrėžimas: (pagalbos į namus + stacionarios socialinės globos gavėjai) / gyvi senjorai * 1000.
"""
from __future__ import annotations

import numpy as np

from .config import NORM_MIN, NORM_TARGET, SCORE_WEIGHTS


def aggregate(series: dict, p) -> dict:
    a = {k: np.asarray(v, float) for k, v in series.items()}
    w = slice(-p.window, None)
    alive = np.maximum(a["alive"][w], 1.0)
    need = np.maximum(a["need"][w], 1.0)
    recip = np.maximum(a["recip"][w], 1.0)
    ind = {
        "R1": float(np.mean(a["recip"][w] / alive) * 1000),            # socialinių paslaugų išvystymas
        "R1_home": float(np.mean(a["home"][w] / alive) * 1000),
        "R1_inst": float(np.mean(a["inst"][w] / alive) * 1000),
        "R1b": float(np.mean(a["aspn"][w] / alive) * 100),             # ASPN aprėptis, %
        "R2": float(np.mean(1.0 - a["unmet"][w] / need) * 100),        # patenkintų poreikių dalis, %
        "R3_queue": float(np.mean(a["queue"][w] / alive) * 1000),      # laukiantys / 1000 senjorų
        "R3_wait": float(np.mean(a["wait_mean"][w])),                  # vid. laukimo trukmė, mėn.
        "R4": float(np.mean(a["unmet"][w] / alive) * 100),             # globos skurdas, % visų senjorų
        "R4_need": float(np.mean(a["unmet"][w] / need) * 100),         # globos skurdas, % turinčių poreikį
        "R5_B": float(np.mean(a["B_mean"][w])),
        "R5_share": float(np.mean(a["burned"][w]) * 100),              # šeimų, viršijusių ribą, %
        "R6": float(a["deaths"][w].sum() / np.mean(alive) * 1000),     # mirčių / 1000 senjorų per metus
        "R7": float(a["alive"][-1]),
        "R8": float(np.mean(a["inst"][w] / alive) * 100),              # institucionalizacijos dalis, %
        "R10": float(np.mean(a["spend"][w] / np.maximum(a["budget"][w], 1e-9)) * 100),
        "R11": float(np.mean(a["spend"][w] / recip)),                  # vieno gavėjo kaina / mėn.
        "respite_per_1000": float(np.mean(a["respite"][w] / alive) * 1000),
        "maxload_social": float(a["maxload_social"][w].max()),
    }
    return ind


def score(ind: dict, weights: dict = SCORE_WEIGHTS) -> float:
    """Kokybės funkcija (didesnė = geriau; maksimumas 0). Ją reikia pagrįsti ir aprašyti darbe."""
    r1 = ind["R1"]
    pen = (weights["below_min"] * max(0.0, (NORM_MIN - r1) / NORM_MIN)
           + weights["below_target"] * max(0.0, (NORM_TARGET - r1) / NORM_TARGET)
           + weights["poverty"] * ind["R4_need"] / 100.0
           + weights["burnout"] * ind["R5_share"] / 100.0)
    return -float(pen)
