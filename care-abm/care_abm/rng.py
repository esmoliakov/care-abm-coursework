"""Atsitiktinių skaičių srautai: PCG64 + SeedSequence.spawn (2.1 sk.)."""
from __future__ import annotations

import numpy as np

# Vienas nepriklausomas srautas kiekvienam procesui / agentų tipui.
STREAMS = ("init", "demography", "need", "family", "allocation", "entry")


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.Generator(np.random.PCG64(c)) for name, c in zip(STREAMS, children)}


def sim_seed(base_seed: int, repeat: int, inner: int) -> int:
    """Simuliacijos sėkla. Nepriklauso nuo scenarijaus => bendri atsitiktiniai skaičiai (CRN)."""
    return int(np.random.SeedSequence([base_seed, repeat, inner]).generate_state(1)[0])


def side_rng(base_seed: int, repeat: int, tag: int) -> np.random.Generator:
    """Atskiras srautas tyrėjams (pvz., atsitiktinė paieška), kad netrikdytų simuliacijos srautų."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([base_seed, repeat, 90_000 + tag])))
