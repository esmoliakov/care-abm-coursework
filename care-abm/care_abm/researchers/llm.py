"""DKM tyrėjas (2.2 sk.): vaidmuo, trumpalaikė / ilgalaikė atmintis, refleksija, įrankiai,
JSON atsakymas, formato / ribų tikrinimas, žetonų ir laiko matavimas.

Ablacijos: long_memory=False (S8), reflection=False (S9).
Įrankiai: palyginti_su_normatyvais() ir gauti_istorija() įvykdomi kode, jų rezultatas įterpiamas
į užklausą; tikrinti_pasiulyma() taikomas po atsakymo (su `max_retries` pakartojimų formato klaidai).
"""
from __future__ import annotations

from ..config import BOUNDS, NORM_MIN, NORM_TARGET
from ..rng import sim_seed
from .base import Proposal, Researcher
from .tools import (KEY2LT, gauti_istorija, palyginti_su_normatyvais, pagrindimo_tikslumas,
                    tikrinti_pasiulyma)

SYSTEM = {
    "lt": (
        "Esi Lietuvos socialinės politikos ekspertas (Auto-Research agentas), analizuojantis ilgalaikės "
        "priežiūros sistemą savivaldybėje.\n"
        "SADM normatyvai (socialinių paslaugų gavėjai 1000-iui senjorų): minimalus 61, siektinas 107.\n"
        "Savivaldybės biudžetas fiksuotas; svertai jį tik perskirsto. Politikos svertai (per iteraciją gali "
        "keisti 1–3):\n"
        "- ambulatorines_dalis (0.10–0.90): biudžeto dalis pagalbai į namus; likusi dalis – stacionariai globai "
        "(viena stacionari vieta kainuoja kelis kartus brangiau nei pagalba į namus);\n"
        "- max_darbuotojo_kruvis (5–20): didžiausias klientų skaičius vienam socialiniam darbuotojui;\n"
        "- atokvepio_prioritetas (0.0–0.5): biudžeto dalis laikino atokvėpio paslaugoms (šeimų perdegimo "
        "prevencija).\n"
        "Tikslas: pasiekti bent minimalų normatyvą (geriau – siektiną), kartu mažinti globos skurdą "
        "(nepatenkintus poreikius) ir šeimų perdegimą. Didesnis gavėjų skaičius nėra tikslas, jei didėja "
        "globos skurdas.\n"
        "Atsakyk TIK vienu JSON objektu, be jokio kito teksto, tokiu formatu:\n"
        '{REFL}"pakeitimai": {"<svertas>": <skaičius>}, "pagrindimas": "<trumpas pagrindimas, cituojantis '
        'tik pateiktus skaičius>"}'
    ),
    "en": (
        "You are an expert in Lithuanian social policy (Auto-Research agent) analysing the long-term care "
        "system of a municipality.\n"
        "Ministry norms (social service recipients per 1000 older persons): minimum 61, target 107.\n"
        "The municipal budget is fixed; the levers only reallocate it. Policy levers (change 1-3 per iteration):\n"
        "- ambulatorines_dalis (0.10-0.90): share of the budget for home help; the rest funds residential care "
        "(one residential place costs several times more than home help);\n"
        "- max_darbuotojo_kruvis (5-20): maximum number of clients per social worker;\n"
        "- atokvepio_prioritetas (0.0-0.5): share of the budget for respite care (carer burnout prevention).\n"
        "Goal: reach at least the minimum norm (preferably the target) while reducing care poverty (unmet "
        "needs) and family burnout. More recipients is not a goal if care poverty grows.\n"
        "Reply ONLY with one JSON object and no other text, in this format:\n"
        '{REFL}"pakeitimai": {"<lever>": <number>}, "pagrindimas": "<short justification citing only the '
        'numbers provided>"}'
    ),
}
REFL_FIELD = {"lt": '"refleksija": "<ar ankstesnis pakeitimas pagerino rodiklius>", ',
              "en": '"refleksija": "<did the previous change improve the indicators>", '}

T = {
    "lt": dict(
        it="Iteracija {k}/{n}.", cur="Dabartiniai svertai: ambulatorines_dalis={a:.2f}, "
        "max_darbuotojo_kruvis={l:.0f}, atokvepio_prioritetas={r:.2f}.",
        head="Rodikliai (vidurkis iš {m} simuliacijų, paskutiniai 12 mėn.):",
        r1="- R1 paslaugų išvystymas: {R1:.1f} gavėjų / 1000 senjorų",
        r2="- Patenkintų poreikių dalis: {R2:.1f} %",
        r4="- Globos skurdas (dalis turinčiųjų poreikį): {R4_need:.1f} %",
        q="- Laukiančiųjų eilė: {R3_queue:.1f} / 1000 senjorų, vid. laukimas {R3_wait:.1f} mėn.",
        b="- Šeimų perdegimas: vid. B_f={R5_B:.2f}; viršijusių 0.8 ribą: {R5_share:.1f} %",
        i="- Institucionalizacijos dalis: {R8:.1f} %", bu="- Biudžeto panaudojimas: {R10:.1f} %",
        tool="Įrankis palyginti_su_normatyvais: skirtumas iki 61 = {d61:+.1f}; iki 107 = {d107:+.1f}.",
        hist="Ankstesnių iteracijų istorija (iteracija | ambulatorines_dalis | max_darbuotojo_kruvis | "
             "atokvepio_prioritetas | R1 | globos skurdas % | perdegimas % | įvertis | tolesnis pakeitimas):",
        refl="Pirmiausia įvertink (laukas „refleksija“), ar ankstesnis pakeitimas pagerino rodiklius.",
        ask="Pasiūlyk pakeitimą JSON formatu.",
        err="Klaida: {e}. Atsakyk TIK teisingu JSON objektu nurodytu formatu."),
    "en": dict(
        it="Iteration {k}/{n}.", cur="Current levers: ambulatorines_dalis={a:.2f}, "
        "max_darbuotojo_kruvis={l:.0f}, atokvepio_prioritetas={r:.2f}.",
        head="Indicators (mean of {m} simulations, last 12 months):",
        r1="- R1 service development: {R1:.1f} recipients / 1000 older persons",
        r2="- Share of needs met: {R2:.1f} %",
        r4="- Care poverty (share of those with a need): {R4_need:.1f} %",
        q="- Waiting list: {R3_queue:.1f} / 1000 older persons, mean wait {R3_wait:.1f} months",
        b="- Family burnout: mean B_f={R5_B:.2f}; above the 0.8 threshold: {R5_share:.1f} %",
        i="- Institutionalisation share: {R8:.1f} %", bu="- Budget utilisation: {R10:.1f} %",
        tool="Tool palyginti_su_normatyvais: gap to 61 = {d61:+.1f}; to 107 = {d107:+.1f}.",
        hist="History of previous iterations (iteration | ambulatorines_dalis | max_darbuotojo_kruvis | "
             "atokvepio_prioritetas | R1 | care poverty % | burnout % | score | next change):",
        refl="First assess (field \"refleksija\") whether the previous change improved the indicators.",
        ask="Propose a change in JSON format.",
        err="Error: {e}. Reply ONLY with a valid JSON object in the specified format."),
}


class LLMResearcher(Researcher):
    def __init__(self, client, lang="lt", long_memory=True, reflection=True, memory_window=5,
                 max_retries=1, base_seed=0, repeat=0, name="llm"):
        self.client, self.lang = client, lang
        self.long_memory, self.reflection = long_memory, reflection
        self.window, self.max_retries = memory_window, max_retries
        self.base_seed, self.repeat = base_seed, repeat
        self.name = name

    # ------------------------------------------------------------------ užklausos
    def _system(self):
        return SYSTEM[self.lang].replace("{REFL}", REFL_FIELD[self.lang] if self.reflection else "")

    def _row(self, h):
        th = h["theta"]
        nxt = ", ".join(f"{KEY2LT[k]}={v:g}" for k, v in h.get("change", {}).items()) or "–"
        return (f"{h['k']} | {th['alpha_amb']:.2f} | {th['l_max']:.0f} | {th['respite']:.2f} | "
                f"{h['ind']['R1']:.1f} | {h['ind']['R4_need']:.1f} | {h['ind']['R5_share']:.1f} | "
                f"{h['score']:.3f} | {nxt}")

    def _user(self, k, n, theta, ind, history, ctx):
        t = T[self.lang]
        cmp_ = palyginti_su_normatyvais(ind["R1"])
        lines = [t["it"].format(k=k, n=n),
                 t["cur"].format(a=theta["alpha_amb"], l=theta["l_max"], r=theta["respite"]),
                 t["head"].format(m=ctx["m"]), t["r1"].format(**ind), t["r2"].format(**ind),
                 t["r4"].format(**ind), t["q"].format(**ind), t["b"].format(**ind),
                 t["i"].format(**ind), t["bu"].format(**ind),
                 t["tool"].format(d61=cmp_["skirtumas_iki_61"], d107=cmp_["skirtumas_iki_107"])]
        past = history[:-1]                                   # paskutinis įrašas = dabartinė iteracija
        rows = gauti_istorija(past, self.window if self.long_memory else 1)
        if rows:
            lines.append(t["hist"])
            lines += [self._row(h) for h in rows]
        if self.reflection and rows:
            lines.append(t["refl"])
        lines.append(t["ask"])
        return "\n".join(lines)

    def _allowed_numbers(self, k, n, theta, ind, ctx):
        nums = [NORM_MIN, NORM_TARGET, 0.8, 12, 36, 100, 1000, float(k), float(n), float(ctx["m"])]
        nums += list(theta.values()) + [v for v in ind.values() if isinstance(v, (int, float))]
        c = palyginti_su_normatyvais(ind["R1"])
        nums += [abs(c["skirtumas_iki_61"]), abs(c["skirtumas_iki_107"]), c["skirtumas_iki_61"], c["skirtumas_iki_107"]]
        return nums

    # ------------------------------------------------------------------ pagrindinis žingsnis
    def propose(self, k, n_iter, theta, ind, history, ctx):
        messages = [{"role": "system", "content": self._system()},
                    {"role": "user", "content": self._user(k, n_iter, theta, ind, history, ctx)}]
        meta = dict(requests=0, format_errors=0, bound_violations=0, api_errors=0,
                    prompt_tokens=0, completion_tokens=0, latency=0.0, just_ok=0, just_n=0,
                    reflection="", justification="", error=None)
        seed = sim_seed(self.base_seed, self.repeat, 50_000 + k)
        attempts = 0
        while True:
            res = self.client.chat(messages, seed=seed + attempts)
            meta["requests"] += 1
            meta["latency"] += res.latency
            meta["prompt_tokens"] += res.prompt_tokens
            meta["completion_tokens"] += res.completion_tokens
            if res.error:
                meta["api_errors"] += 1
                meta["error"] = res.error
                return Proposal(dict(theta), False, meta)
            changes, etype, msg, obj = tikrinti_pasiulyma(res.text)
            if obj:
                meta["reflection"] = str(obj.get("refleksija", ""))[:500]
                meta["justification"] = str(obj.get("pagrindimas", ""))[:800]
                ok, tot = pagrindimo_tikslumas(meta["justification"],
                                               self._allowed_numbers(k, n_iter, theta, ind, ctx))
                meta["just_ok"], meta["just_n"] = ok, tot
            if changes is not None:
                new = dict(theta)
                new.update(changes)
                new["l_max"] = float(round(new["l_max"]))
                meta["change"] = changes
                return Proposal(new, True, meta)
            meta["error"] = msg
            if etype == "bounds":                    # 1 algoritmas: už ribų => paliekami ankstesni parametrai
                meta["bound_violations"] += 1
                return Proposal(dict(theta), False, meta)
            meta["format_errors"] += 1               # formato klaida => pakartojimas (kol leidžiama)
            attempts += 1
            if attempts > self.max_retries:
                return Proposal(dict(theta), False, meta)
            messages += [{"role": "assistant", "content": res.text},
                         {"role": "user", "content": T[self.lang]["err"].format(e=msg)}]
