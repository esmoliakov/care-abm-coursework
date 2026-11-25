"""Tyrėjo sąsaja: propose(...) -> Proposal."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Proposal:
    theta: dict                       # nauji svertai (jei nepriimta – ankstesni)
    accepted: bool = True
    meta: dict = field(default_factory=dict)


class Researcher:
    name = "base"
    iterative = True                  # False => nevykdomas iteracijų ciklas (S1, S2)

    def propose(self, k: int, n_iter: int, theta: dict, ind: dict, history: list, ctx: dict) -> Proposal:
        raise NotImplementedError
