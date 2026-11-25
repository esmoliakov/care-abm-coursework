"""Eksperimentų scenarijai (2.4 sk. 6 lentelė) + papildomi kontroliniai scenarijai.

Papildymai, ištaisantys plano spragas:
  S4b       – taisyklėmis grįstas tyrėjas su visais trimis svertais (sąžiningesnis palyginimas)
  G1 / G2   – visas tinklelis (80 svertų kombinacijų): tikrasis optimumas ir regret skaičiavimui
  S11_*/S12_* – šokai su kontroliniais tyrėjais (be jų negalima teigti, kad DKM geriau prisitaiko)
  S13/S14   – jautrumo analizė (kai tyrėjas nenaudojamas); tyrėjo atsparumą tikrina *_rule / *_llm variantai
Bet kurį scenarijų galima paleisti su kitais parametrais: --set family_share=0.4
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    id: str
    profile: str                       # "low" | "high"
    researcher: str                    # fixed | random | rule | rule_full | llm | grid
    repeats: int
    model: str | None = None           # DKM pavadinimas iš configs/models.yaml (llm)
    overrides: dict = field(default_factory=dict)
    llm_opts: dict = field(default_factory=dict)
    desc: str = ""


def build_registry() -> dict[str, Scenario]:
    R: dict[str, Scenario] = {}

    def add(**kw):
        s = Scenario(**kw)
        R[s.id] = s

    add(id="S1", profile="low", researcher="fixed", repeats=100, desc="Bazinis (Vilniaus r.), be tyrėjo")
    add(id="S2", profile="high", researcher="fixed", repeats=100, desc="Kontrolinis profilis (Šiaulių r.)")
    add(id="S3", profile="low", researcher="random", repeats=50, desc="Atsitiktinė paieška")
    add(id="S4", profile="low", researcher="rule", repeats=50, desc="Taisyklė: R1<61 => didinti alpha_amb")
    add(id="S4b", profile="low", researcher="rule_full", repeats=50, desc="Taisyklė su 3 svertais")
    add(id="S5", profile="low", researcher="llm", model="DKM-A", repeats=30, desc="DKM-A")
    add(id="S6", profile="low", researcher="llm", model="DKM-B", repeats=30, desc="DKM-B")
    add(id="S7", profile="low", researcher="llm", model="DKM-C", repeats=30, desc="DKM-C (komercinis)")
    add(id="S8", profile="low", researcher="llm", repeats=30, llm_opts={"long_memory": False},
        desc="Geriausias DKM, be ilgalaikės atminties (--model)")
    add(id="S9", profile="low", researcher="llm", repeats=30, llm_opts={"reflection": False},
        desc="Geriausias DKM, be refleksijos (--model)")
    add(id="S10", profile="high", researcher="llm", repeats=30, desc="Aukštas profilis (--model)")
    for sid, ov, d in (("S11", {"shock_family_support": 0.20}, "šeimos pagalba -20 %"),
                       ("S12", {"shock_budget": 0.15}, "biudžetas -15 %")):
        add(id=sid, profile="low", researcher="llm", repeats=30, overrides=ov, desc=f"Šokas: {d} (--model)")
        add(id=f"{sid}_rnd", profile="low", researcher="random", repeats=50, overrides=ov, desc=f"{d}, atsitiktinė")
        add(id=f"{sid}_rule", profile="low", researcher="rule_full", repeats=50, overrides=ov, desc=f"{d}, taisyklė")
        add(id=f"{sid}_fixed", profile="low", researcher="fixed", repeats=50, overrides=ov, desc=f"{d}, be tyrėjo")
    for sid, key, vals in (("S13", "family_share", (0.4, 0.8)), ("S14", "burnout_threshold", (0.7, 0.9))):
        for tag, v in zip("ab", vals):
            add(id=f"{sid}{tag}", profile="low", researcher="fixed", repeats=50, overrides={key: v},
                desc=f"Jautrumas: {key}={v}")
            add(id=f"{sid}{tag}_rule", profile="low", researcher="rule_full", repeats=50, overrides={key: v},
                desc=f"Tyrėjo atsparumas: {key}={v}, taisyklė")
            add(id=f"{sid}{tag}_llm", profile="low", researcher="llm", repeats=30, overrides={key: v},
                desc=f"Tyrėjo atsparumas: {key}={v}, DKM (--model)")
    add(id="S15", profile="low", researcher="fixed", repeats=50, overrides={"h_thr_low": 0.2, "h_thr_high": 0.5},
        desc="Jautrumas: sveikatos slenksčiai -0.1")
    add(id="S15b", profile="low", researcher="fixed", repeats=50, overrides={"h_thr_low": 0.4, "h_thr_high": 0.7},
        desc="Jautrumas: sveikatos slenksčiai +0.1")
    add(id="S16", profile="low", researcher="fixed", repeats=50, overrides={"n_seniors": 5000, "n_workers": 375,
        "n_social_workers": 150}, desc="Mastelio patikra: N=5000 (biudžetas nesikeičia vienam senjorui)")
    add(id="G1", profile="low", researcher="grid", repeats=1, desc="Tinklelis, žemas profilis")
    add(id="G2", profile="high", researcher="grid", repeats=1, desc="Tinklelis, aukštas profilis")
    return R