"""LLM klientas: viena OpenAI-suderinama sąsaja visiems modeliams (3.1 sk.).

Veikia su: OpenAI (GPT), vLLM / Ollama (atviri modeliai HPC ar lokaliai), OpenRouter, Together, Groq,
Gemini (OpenAI-compat. endpoint), Anthropic (OpenAI SDK compat. endpoint) – keičiasi tik base_url ir model.
`provider: mock` – testavimui be API (nemokamai, deterministiškas).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LLMResult:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    error: str | None = None
    estimated_tokens: bool = False


class OpenAICompatClient:
    def __init__(self, cfg: dict):
        from openai import OpenAI                      # importuojama tik čia
        self.cfg = cfg
        key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), cfg.get("api_key", "EMPTY"))
        self.client = OpenAI(base_url=cfg.get("base_url"), api_key=key,
                             timeout=cfg.get("timeout", 180), max_retries=0)
        self.json_mode = cfg.get("json_mode", "json_object")      # "json_object" | "none"

    def chat(self, messages: list[dict], seed: int | None = None) -> LLMResult:
        c = self.cfg
        kw = dict(model=c["model"], messages=messages)
        if c.get("send_temperature", True) and c.get("temperature") is not None:
            kw["temperature"] = c["temperature"]
        if c.get("send_top_p", True) and c.get("top_p") is not None:
            kw["top_p"] = c["top_p"]
        # GPT-5 / o-serijos modeliai: token_param: max_completion_tokens, send_temperature: false
        kw[c.get("token_param", "max_tokens")] = c.get("max_tokens", 500)
        if seed is not None and c.get("use_seed", True):
            kw["seed"] = int(seed) % (2 ** 31 - 1)
        if self.json_mode == "json_object":
            kw["response_format"] = {"type": "json_object"}
        if c.get("extra_body"):
            kw["extra_body"] = c["extra_body"]
        kw.update(c.get("extra_params", {}))

        last_err = None
        for attempt in range(int(c.get("api_retries", 5))):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**kw)
                dt = time.time() - t0
                text = (resp.choices[0].message.content or "")
                u = getattr(resp, "usage", None)
                if u is not None and getattr(u, "prompt_tokens", None) is not None:
                    return LLMResult(text, u.prompt_tokens, u.completion_tokens, dt)
                n_in = sum(len(m["content"]) for m in messages) // 4
                return LLMResult(text, n_in, len(text) // 4, dt, estimated_tokens=True)
            except Exception as e:                                         # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                if "response_format" in str(e) and "response_format" in kw:
                    self.json_mode = "none"                                # serveris nepalaiko JSON režimo
                    kw.pop("response_format")
                    continue
                time.sleep(min(2 ** attempt, 30))
        return LLMResult(error=last_err)


class MockClient:
    """Imituoja DKM: paprasta euristika + kartais sugadintas JSON / ribų pažeidimas (T3, T4 testams)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.p_bad_json = cfg.get("p_bad_json", 0.10)
        self.p_out_of_bounds = cfg.get("p_out_of_bounds", 0.05)

    def chat(self, messages: list[dict], seed: int | None = None) -> LLMResult:
        user = messages[-1]["content"]
        h = int(hashlib.sha256((user + str(seed)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        num = lambda pat, default: float(re.search(pat, user).group(1)) if re.search(pat, user) else default  # noqa: E731
        r1 = num(r"R1[^\n]*?:\s*(-?\d+(?:\.\d+)?)", 50.0)
        pov = num(r"Globos skurdas[^\n]*?:\s*(-?\d+(?:\.\d+)?)", 20.0)
        alpha = num(r"ambulatorines_dalis=(\d+(?:\.\d+)?)", 0.6)
        resp = num(r"atokvepio_prioritetas=(\d+(?:\.\d+)?)", 0.1)
        u = rng.random()
        if u < self.p_bad_json:
            text = "Siūlau didinti ambulatorinę dalį, nes rodiklis per žemas."          # ne JSON
        else:
            if r1 < 61:
                ch = {"ambulatorines_dalis": round(min(0.9, alpha + 0.1), 2)}
            elif pov > 15:
                ch = {"atokvepio_prioritetas": round(min(0.5, resp + 0.05), 2)}
            else:
                ch = {"max_darbuotojo_kruvis": 15}
            if rng.random() < self.p_out_of_bounds:
                ch = {"ambulatorines_dalis": 1.2}
            text = json.dumps({"refleksija": "Ankstesnis pakeitimas tikriausiai pagerino rodiklius.",
                               "pakeitimai": ch,
                               "pagrindimas": f"Rodiklis ({r1:.1f}/1000) lyginamas su normatyvu 61; "
                                              f"globos skurdas {pov:.1f} %."}, ensure_ascii=False)
        n_in = sum(len(m["content"]) for m in messages) // 4
        return LLMResult(text, n_in, len(text) // 4, 0.01, estimated_tokens=True)


def make_client(cfg: dict):
    if cfg.get("provider") == "mock":
        return MockClient(cfg)
    return OpenAICompatClient(cfg)


_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_dotenv(path: str = _ENV_FILE) -> None:
    """Nuskaito projekto šaknies .env (KEY=reikšmė). Jau nustatytų aplinkos kintamųjų nekeičia."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        else:
            v = v.split(" #", 1)[0].strip()          # komentaras eilutės gale
        os.environ.setdefault(k, v)


def _expand(value, name: str):
    """${KINTAMASIS} -> reikšmė iš aplinkos / .env."""
    if isinstance(value, str):
        def sub(m):
            var = m.group(1)
            if var not in os.environ or os.environ[var] == "":
                raise KeyError(f"Modeliui {name} reikia aplinkos kintamojo {var} (įrašykite į .env)")
            return os.environ[var]
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, name) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, name) for v in value]
    return value


def load_model_cfg(path: str, name: str) -> dict:
    import yaml
    load_dotenv()
    d = yaml.safe_load(open(path, encoding="utf-8"))
    cfg = dict(d.get("generation", {}))
    cfg.update(_expand(d["models"][name], name))
    cfg["name"] = name
    return cfg