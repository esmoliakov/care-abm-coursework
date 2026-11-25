"""Parametrai, normatyvai, politikos svertų ribos ir profilių biudžetai.

"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace, fields
from pathlib import Path
from typing import Any, Optional

# --- SADM normatyvai (gavėjai / 1000 senjorų) ir kalibravimo taikiniai -------------------
NORM_MIN = 61.0
NORM_TARGET = 107.0
TARGET_R1 = {"low": 41.9, "high": 312.7}          # Vilniaus r. ir Šiaulių r., 2025 m.

# --- Politikos svertai (tyrėjo valdomi) ---------------------------------------------------
POLICY_KEYS = ("alpha_amb", "l_max", "respite")
BOUNDS = {"alpha_amb": (0.10, 0.90), "l_max": (5.0, 20.0), "respite": (0.0, 0.5)}
# Pradinė (neoptimizuota) politika: daug institucinės globos, mažas darbuotojų krūvis, didelis atokvėpio prioritetas.
# Pasirinkta tyčia, kad tyrėjui liktų ką optimizuoti (tinklelio patikra G1/G2 parodo paviršių).
DEFAULT_THETA = {"alpha_amb": 0.35, "l_max": 8.0, "respite": 0.20}

# Tyrėjo kokybės (tikslo) funkcija. Pasiūlymas problemai, kad R1 vienas netinka tikslui:
# bauda už nepasiektą minimumą / siektiną normatyvą + globos skurdas + šeimų perdegimas.
SCORE_WEIGHTS = {"below_min": 2.0, "below_target": 1.0, "poverty": 1.0, "burnout": 0.5}

_PROFILE_FILE = Path(__file__).resolve().parent.parent / "configs" / "profiles.json"


def profile_budgets() -> dict:
    """Profilių (low/high) savivaldybės biudžetas vienam senjorui per mėnesį (po kalibravimo)."""
    if not _PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Nerastas {_PROFILE_FILE}. Pirmiausia paleiskite: python -m care_abm.calibrate")
    return json.loads(_PROFILE_FILE.read_text())["budget_per_senior"]


@dataclass
class SimParams:
    # --- mastelis (2.1 sk. lentelė) ---
    n_seniors: int = 2000
    family_share: float = 0.60            # šeimos pagalbos tinklo dalis (S13: 0.4 / 0.8)
    n_workers: int = 150
    n_social_workers: int = 60            # prielaida; 60, kad aukšto profilio taikinys (312,7) būtų pasiekiamas (lubos su 50 – 304)
    months: int = 36
    window: int = 12                      # rodikliai = paskutinių 12 mėn. vidurkis

    # --- sveikata ir poreikis ---
    h_beta_a: float = 3.0                 # pradinė H ~ Beta(a, b)  
    h_beta_b: float = 2.0
    h_thr_low: float = 0.3                # < 0.3 stacionarus poreikis
    h_thr_high: float = 0.6               # 0.3–0.6 ambulatorinis poreikis
    p_decline_base: float = 0.20          # metinė sveikatos šuolio tikimybė 65 m
    p_decline_slope: float = 0.008        # + už kiekvienus metus virš 65
    delta_mu: float = 0.08                # šuolio dydis ~ N(mu, sigma^2), apkarpytas
    delta_sigma: float = 0.05

    # --- demografija ---
    beta: float = 1.5                     # sveikatos poveikis mirtingumui 
    gompertz_a0: float = 0.017            # nenaudojami
    gompertz_b: float = 0.09
    life_table_csv: Optional[str] = "configs/life_table.csv" # age, q metine mirties tikimybe
    entrant_rate_annual: float = 0.062     # E_t / S_t
    p_emigration_annual: float = 0.013     # šeimos nario emigracija 

    # --- šeima ir perdegimas ---
    burnout_threshold: float = 0.8        # S14: 0.7 / 0.9
    bg_ambulatory: float = 0.020          # B_f augimas per mėn., ambulatorinis poreikis
    bg_stationary: float = 0.060          # B_f augimas per mėn., stacionarus poreikis
    b_recover: float = 0.010              # B_f mažėjimas, kai poreikio nėra
    relief_formal: float = 0.50           # augimo daugiklis, kai gauna formalią socialinę paslaugą
    relief_aspn: float = 0.75             # augimo daugiklis, kai gauna ASPN
    respite_recovery: float = 0.06        # B_f sumažėjimas gavus laikino atokvėpio paslaugą
    p_apply_with_family: float = 0.60     # prašymo teikimo tikimybė, kai šeima pajėgi prižiūrėti

    # --- kainos ir ASPN ---
    cost_home: float = 1.0
    cost_inst: float = 4.0                # 4 kartus brangiau nei ambulatorinė pagalba
    cost_respite: float = 0.5
    aspn_capacity_share: float = 0.062    # OECD 2022: ~6,2 % (nacionalinis intensyvumas)

    # --- savivaldybės profilis ---
    budget_per_senior: float = 0.0        # užpildoma iš profiles.json

    # --- šokiai (S11, S12) ---
    shock_month: int = 12
    shock_family_support: float = 0.0     # dalis šeimų, prarandamų šoko metu
    shock_budget: float = 0.0             # biudžeto sumažėjimas nuo šoko mėnesio


def with_overrides(p: SimParams, overrides: dict[str, Any]) -> SimParams:
    names = {f.name for f in fields(SimParams)}
    for k in overrides:
        if k not in names:
            raise KeyError(f"Nežinomas parametras: {k}")
    return replace(p, **overrides)


def params_for_profile(profile: str, overrides: Optional[dict] = None) -> SimParams:
    p = SimParams(budget_per_senior=profile_budgets()[profile])
    return with_overrides(p, overrides or {})


def theta_in_bounds(theta: dict) -> bool:
    return all(BOUNDS[k][0] <= theta[k] <= BOUNDS[k][1] for k in POLICY_KEYS)