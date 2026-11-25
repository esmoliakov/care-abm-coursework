"""Tyrėjo įrankiai (skaičiavimus atlieka kodas, ne kalbos modelis): palyginimas su normatyvais,
istorija, pasiūlymo tikrinimas pagal leistinas ribas."""
from __future__ import annotations

import json
import re

from ..config import BOUNDS, NORM_MIN, NORM_TARGET, POLICY_KEYS

LT2KEY = {"ambulatorines_dalis": "alpha_amb",
          "max_darbuotojo_kruvis": "l_max",
          "atokvepio_prioritetas": "respite"}
KEY2LT = {v: k for k, v in LT2KEY.items()}


def palyginti_su_normatyvais(r1: float) -> dict:
    return {"skirtumas_iki_61": round(NORM_MIN - r1, 2), "skirtumas_iki_107": round(NORM_TARGET - r1, 2),
            "min_pasiektas": r1 >= NORM_MIN, "siektinas_pasiektas": r1 >= NORM_TARGET}


def gauti_istorija(history: list, k: int) -> list:
    return history[-k:] if k > 0 else []


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("JSON objektas nerastas")
    return json.loads(t[i:j + 1])


def tikrinti_pasiulyma(text: str):
    """Grąžina (changes|None, klaidos_tipas|None, pranešimas, obj|None).
    klaidos_tipas: 'format' (T3) arba 'bounds' (T4)."""
    try:
        obj = extract_json(text)
    except Exception as e:                                           # noqa: BLE001
        return None, "format", f"neteisingas JSON: {e}", None
    if not isinstance(obj, dict):
        return None, "format", "atsakymas nėra JSON objektas", None
    changes = obj.get("pakeitimai")
    if changes is None and "parametras" in obj:                      # suderinamumas su 2.2 sk. formatu
        changes = {obj["parametras"]: obj.get("nauja_reiksme")}
    if not isinstance(changes, dict) or not changes:
        return None, "format", "trūksta lauko 'pakeitimai' (objektas su svertais)", obj
    out = {}
    for k, v in changes.items():
        if k not in LT2KEY:
            return None, "format", f"nežinomas svertas '{k}'", obj
        try:
            out[LT2KEY[k]] = float(v)
        except (TypeError, ValueError):
            return None, "format", f"reikšmė '{v}' nėra skaičius", obj
    for key, v in out.items():
        lo, hi = BOUNDS[key]
        if not (lo <= v <= hi):
            return None, "bounds", f"{KEY2LT[key]}={v} už leistinų ribų [{lo}, {hi}]", obj
    return out, None, "", obj


_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:[.,]\d+)?")   # „R1“, „B_f2“ neskaičiuojami kaip skaičiai


def pagrindimo_tikslumas(text: str, allowed: list[float]):
    """T7: pagrindime minimų skaičių, sutampančių su pateiktais rodikliais, dalis. (sutampa, iš viso)"""
    nums = [float(x.replace(",", ".")) for x in _NUM.findall(text or "")]
    ok = 0
    for x in nums:
        if any(abs(x - y) <= max(0.06, 0.01 * abs(y)) for y in allowed):
            ok += 1
    return ok, len(nums)
