"""Greiti patikrinimai: python -m pytest -q"""
import numpy as np
import pytest

from care_abm.config import BOUNDS, DEFAULT_THETA, SimParams, params_for_profile
from care_abm.model import run_one
from care_abm.researchers.tools import pagrindimo_tikslumas, tikrinti_pasiulyma
from care_abm.run_experiment import run_repeat

A = dict(base_seed=1, iterations=3, inner=2, eval=3, lang="lt", memory_window=5, max_retries=1,
         overrides={}, force=True, crn_across_iterations=True, grid_repeats=1)


def test_determinism():
    p = params_for_profile("low")
    a, b = run_one(p, DEFAULT_THETA, 7), run_one(p, DEFAULT_THETA, 7)
    c = run_one(p, DEFAULT_THETA, 8)
    assert a == b            # ta pati sėkla => identiškas rezultatas
    assert a != c            # kita sėkla => kitoks rezultatas


def test_budget_and_worker_constraints():
    for prof in ("low", "high"):
        p = params_for_profile(prof)
        _, s = run_one(p, DEFAULT_THETA, 3, return_series=True)
        assert all(sp <= bd + 1e-9 for sp, bd in zip(s["spend"], s["budget"]))
        assert max(s["maxload_social"]) <= DEFAULT_THETA["l_max"]


def test_profiles_ordering_and_targets():
    lo = np.mean([run_one(params_for_profile("low"), DEFAULT_THETA, s)["R1"] for s in range(8)])
    hi = np.mean([run_one(params_for_profile("high"), DEFAULT_THETA, s)["R1"] for s in range(8)])
    assert abs(lo - 41.9) < 3 and abs(hi - 312.7) < 15


def test_common_random_numbers_demography_independent_of_policy():
    p = params_for_profile("low")
    _, s1 = run_one(p, DEFAULT_THETA, 5, return_series=True)
    _, s2 = run_one(p, {"alpha_amb": 0.9, "l_max": 20, "respite": 0.0}, 5, return_series=True)
    assert s1["alive"] == s2["alive"]


def test_shocks_change_outcomes():
    base = run_one(params_for_profile("low"), DEFAULT_THETA, 2)
    cut = run_one(params_for_profile("low", {"shock_budget": 0.5}), DEFAULT_THETA, 2)
    assert cut["R1"] < base["R1"]


def test_parser():
    ok, et, _, _ = tikrinti_pasiulyma('{"pakeitimai": {"ambulatorines_dalis": 0.7}, "pagrindimas": "x"}')
    assert ok == {"alpha_amb": 0.7} and et is None
    assert tikrinti_pasiulyma("ne json")[1] == "format"
    assert tikrinti_pasiulyma('{"pakeitimai": {"ambulatorines_dalis": 1.5}}')[1] == "bounds"
    assert tikrinti_pasiulyma('{"parametras": "atokvepio_prioritetas", "nauja_reiksme": 0.3}')[0] == {"respite": 0.3}
    assert tikrinti_pasiulyma('```json\n{"pakeitimai": {"max_darbuotojo_kruvis": 10}}\n```')[0] == {"l_max": 10.0}
    assert pagrindimo_tikslumas("R1 yra 44.5 o normatyvas 61", [44.5, 61]) == (2, 2)


@pytest.mark.parametrize("sid,model", [("S1", None), ("S3", None), ("S4", None), ("S4b", None), ("S5", "MOCK")])
def test_run_repeat_all_researchers(sid, model):
    cfg = {"provider": "mock", "name": "MOCK"} if model else None
    r = run_repeat(sid, 0, A, cfg)
    assert "last" in r["final"] and "best" in r["final"]
    if sid != "S1":
        assert len(r["iterations"]) == 3
        assert all(BOUNDS[k][0] <= v <= BOUNDS[k][1] for it in r["iterations"] for k, v in it["theta"].items())
