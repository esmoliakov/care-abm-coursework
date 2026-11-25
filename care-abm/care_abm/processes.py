"""Mėnesio žingsnio procesai (2.1 sk., 3 skyrius „Procesų apžvalga“).

Seka: 1) demografija, 2) poreikio kaita, 3) neformali šeimos globa, 4) formalių paslaugų priskyrimas
(+ eilė), 5) rodiklių agregavimas (model.py).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- pagalbinės
def monthly_prob(p_annual):
    """(1) lygtis: p_men = 1 - (1 - p_met)^(1/12)."""
    return 1.0 - (1.0 - np.asarray(p_annual, float)) ** (1.0 / 12.0)


def need_from_h(H, lo, hi):
    return np.where(H < lo, 2, np.where(H < hi, 1, 0)).astype(np.int8)


@lru_cache(maxsize=4)
def _load_life_table(path: str):
    import pandas as pd
    df = pd.read_csv(path).sort_values("age")
    return df["age"].to_numpy(float), df["q"].to_numpy(float)


def mu0(p, age):
    """Bazinis metinis mirtingumo hazardas mu_0(a) = -ln(1 - q_a)."""
    if p.life_table_csv:
        path = Path(p.life_table_csv)
        if not path.is_absolute():            # santykinis kelias – projekto šaknies atžvilgiu, ne darbinio katalogo
            path = Path(__file__).resolve().parent.parent / path
        ages, q = _load_life_table(str(path))
        qa = np.interp(age, ages, q)
        return -np.log(1.0 - np.clip(qa, 0, 0.999))
    return np.minimum(p.gompertz_a0 * np.exp(p.gompertz_b * (age - 65.0)), 3.0)  # Gompertz (placeholder)


# ----------------------------------------------------------------------------- inicializacija
def init_population(m):
    p, S, F = m.p, m.S, m.F
    rng = m.rng["init"]
    n = p.n_seniors
    age = 65.0 + np.minimum(rng.gamma(2.0, 6.0, n), 35.0)
    H = rng.beta(p.h_beta_a, p.h_beta_b, n)
    idx = S.add(age, H, need_from_h(H, p.h_thr_low, p.h_thr_high))
    S.u_apply[idx] = rng.random(n)
    k = int(round(p.family_share * n))
    fam = rng.choice(idx, size=k, replace=False)
    F.exists[fam] = True
    F.B[fam] = rng.beta(1.5, 6.0, k)


def add_entrants(m, n_new: int):
    if n_new <= 0:
        return
    p, S, F = m.p, m.S, m.F
    rng = m.rng["entry"]
    age = 65.0 + rng.random(n_new)
    H = rng.beta(p.h_beta_a + 0.5, p.h_beta_b, n_new)        # ką tik pasiekę pensinį amžių – sveikesni
    idx = S.add(age, H, need_from_h(H, p.h_thr_low, p.h_thr_high))
    S.u_apply[idx] = rng.random(n_new)
    has = rng.random(n_new) < p.family_share
    F.exists[idx[has]] = True
    F.B[idx[has]] = rng.beta(1.5, 6.0, int(has.sum()))


# ----------------------------------------------------------------------------- 1. demografija
def demography(m):
    p, S, F = m.p, m.S, m.F
    rng = m.rng["demography"]
    idx = np.flatnonzero(S.alive)
    S.age[idx] += 1.0 / 12.0
    age, H = S.age[idx], S.H[idx]

    # (2) mu_i = mu_0(a) * exp(beta (1-H)), normalizuota amžiaus grupės vidurkiu
    w = np.exp(p.beta * (1.0 - H))
    grp = np.clip(((age - 65.0) // 5).astype(int), 0, 9)
    sums = np.bincount(grp, weights=w, minlength=10)
    cnt = np.bincount(grp, minlength=10)
    mean_w = np.where(cnt > 0, sums / np.maximum(cnt, 1), 1.0)
    mu = mu0(p, age) * w / mean_w[grp]
    p_die = 1.0 - np.exp(-mu / 12.0)                          # (3) lygtis
    die = idx[rng.random(idx.size) < p_die]
    S.alive[die] = False
    S.formal[die] = 0
    S.aspn[die] = False
    S.applicant[die] = False
    S.wait[die] = 0
    F.exists[die] = False
    m.deaths_this_month = int(die.size)

    # šeimos narių emigracija (Bernulio, p_men iš (1))
    fam_idx = np.flatnonzero(F.exists & S.alive)
    left = fam_idx[rng.random(fam_idx.size) < float(monthly_prob(p.p_emigration_annual))]
    F.exists[left] = False

    # nauji senjorai: Puasono, lambda = N0 * (E/S) / 12
    lam = p.n_seniors * p.entrant_rate_annual / 12.0
    add_entrants(m, int(m.rng["entry"].poisson(lam)))


# ----------------------------------------------------------------------------- 2. poreikio kaita
def need_change(m):
    p, S = m.p, m.S
    rng = m.rng["need"]
    idx = np.flatnonzero(S.alive)
    p_ann = np.clip(p.p_decline_base + p.p_decline_slope * (S.age[idx] - 65.0), 0.0, 0.95)
    ev = rng.random(idx.size) < monthly_prob(p_ann)
    delta = np.clip(rng.normal(p.delta_mu, p.delta_sigma, idx.size), 0.0, 1.0)
    S.H[idx] = np.where(ev, np.maximum(0.0, S.H[idx] - delta), S.H[idx])
    S.need[idx] = need_from_h(S.H[idx], p.h_thr_low, p.h_thr_high)


# ----------------------------------------------------------------------------- 3. neformali globa
def informal_care(m):
    p, S, F = m.p, m.S, m.F
    active = F.exists & S.alive & (F.B < p.burnout_threshold)   # perdegę – globą nutraukia visam laikui
    need = S.need
    g = np.where(need == 2, p.bg_stationary, np.where(need == 1, p.bg_ambulatory, -p.b_recover))
    factor = np.ones(S.cap)
    factor[S.formal > 0] *= p.relief_formal
    factor[S.aspn] *= p.relief_aspn
    dB = np.where(g > 0, g * factor, g) - p.respite_recovery * F.resp_prev
    F.B[active] = np.clip(F.B[active] + dB[active], 0.0, 1.0)


def apply_family_shock(m):
    """S11: šeimos pagalbos tinklas sumažėja (dalis šeimų pasitraukia)."""
    p, S, F = m.p, m.S, m.F
    if p.shock_family_support <= 0:
        return
    idx = np.flatnonzero(F.exists & S.alive)
    gone = idx[m.rng["family"].random(idx.size) < p.shock_family_support]
    F.exists[gone] = False


# ----------------------------------------------------------------------------- 4. formalūs paslaugos
def _prioritise(cand, S, fam_ok, rng):
    """Eilės prioritetas: (i) nėra pajėgios šeimos, (ii) stacionarus poreikis, (iii) ilgiau laukia, (iv) atsitiktinai."""
    key = 2 * (~fam_ok[cand]).astype(int) + (S.need[cand] == 2).astype(int)
    order = np.lexsort((rng.random(cand.size), -S.wait[cand], -key))
    return cand[order]


def allocate(m):
    p, S, F = m.p, m.S, m.F
    rng = m.rng["allocation"]
    t = m.month
    alive = S.alive
    fam_ok = F.ok(alive, p.burnout_threshold)
    need_mask = alive & (S.need >= 1)
    applicant = need_mask & ((~fam_ok) | (S.u_apply < p.p_apply_with_family))
    S.formal[~need_mask] = 0
    cap_home, cap_inst, cap_resp = m.M.capacities(t)

    # paslaugos išlaikomos; sumažėjus pajėgumui (šokas) – paskutiniai priimti išvedami
    for code, cap in ((2, cap_inst), (1, cap_home)):
        cur = np.flatnonzero(S.formal == code)
        if cur.size > cap:
            drop = cur[np.argsort(S.admit[cur], kind="stable")[cap:]]
            S.formal[drop] = 0

    # nauji stacionaro priėmimai (tik stacionarus poreikis)
    free = cap_inst - int((S.formal == 2).sum())
    if free > 0:
        cand = np.flatnonzero(applicant & (S.need == 2) & (S.formal != 2))
        take = _prioritise(cand, S, fam_ok, rng)[:free]
        S.formal[take] = 2
        S.admit[take] = t

    # nauji pagalbos į namus gavėjai (stacionaraus poreikio asmenys – kaip „artimiausia galima“ alternatyva)
    free = cap_home - int((S.formal == 1).sum())
    if free > 0:
        cand = np.flatnonzero(applicant & (S.formal == 0))
        take = _prioritise(cand, S, fam_ok, rng)[:free]
        S.formal[take] = 1
        S.admit[take] = t

    # laikino atokvėpio paslauga: šeimoms su didžiausiu perdegimu
    F.resp_prev[:] = False
    if cap_resp > 0:
        cand = np.flatnonzero(F.exists & alive & (F.B < p.burnout_threshold) & (S.need >= 1))
        if cand.size:
            top = cand[np.argsort(-F.B[cand], kind="stable")[:cap_resp]]
            F.resp_prev[top] = True

    # ASPN (sveikatos šaka): nacionalinis intensyvumas, nevaldomas savivaldybės
    cap_aspn = int(round(p.aspn_capacity_share * int(alive.sum())))
    cand = np.flatnonzero(need_mask)
    order = np.lexsort((rng.random(cand.size), -S.need[cand], -S.aspn[cand].astype(int)))
    S.aspn[:] = False
    S.aspn[cand[order[:cap_aspn]]] = True

    # eilė: prašymą pateikę, bet formalios paslaugos negaunantys
    queue = applicant & (S.formal == 0)
    S.wait[queue] += 1
    S.wait[~queue] = 0
    S.applicant[:] = applicant

    m.W.assign(int((S.formal == 1).sum()), int(S.aspn.sum()))