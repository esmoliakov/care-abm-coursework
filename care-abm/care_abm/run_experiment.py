"""Scenarijaus (S1–S16, G1–G2) paleidimas ir rezultatų įrašymas (3.1 sk. run_experiment.py).

Pavyzdžiai:
  python -m care_abm.run_experiment --list
  python -m care_abm.run_experiment --scenario S1 --workers 8
  python -m care_abm.run_experiment --scenario S5 --model DKM-A --workers 4
  python -m care_abm.run_experiment --scenario S5 --model MOCK --repeats 3      # bandymas be API
  python -m care_abm.run_experiment --scenario S3 --start 0 --repeats 10        # SLURM masyvui: --start/--repeats
Rezultatai: results/<scenarijus>[__<modelis>]/rep_XXXX.json (jau esantys praleidžiami => galima tęsti).

Eksperimento ciklas (1 algoritmas, patikslintas): VISADA vykdoma I iteracijų (be ankstyvo stabdymo, kad
R9 – laikas iki 61 ir 107 – būtų matuojamas); gale politika vertinama ant atskirų (held-out) sėklų.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import (DEFAULT_THETA, NORM_MIN, NORM_TARGET, params_for_profile)
from .indicators import score
from .model import evaluate
from .researchers import FixedResearcher, RandomSearchResearcher, RuleBasedResearcher
from .researchers.llm import LLMResearcher
from .rng import side_rng, sim_seed
from .scenarios import build_registry

GRID = {"alpha_amb": [0.1, 0.3, 0.5, 0.7, 0.9], "l_max": [5, 10, 15, 20], "respite": [0.0, 0.15, 0.3, 0.5]}


# ------------------------------------------------------------------------------ tyrėjo sukūrimas
def build_researcher(scn, repeat, a, model_cfg):
    kind = scn.researcher
    if kind == "fixed":
        return FixedResearcher()
    if kind == "random":
        return RandomSearchResearcher(side_rng(a["base_seed"], repeat, 1))
    if kind in ("rule", "rule_full"):
        return RuleBasedResearcher(full=(kind == "rule_full"))
    if kind == "llm":
        from .llm_client import make_client
        opts = dict(scn.llm_opts)
        return LLMResearcher(make_client(model_cfg), lang=a["lang"], long_memory=opts.get("long_memory", True),
                             reflection=opts.get("reflection", True), memory_window=a["memory_window"],
                             max_retries=a["max_retries"], base_seed=a["base_seed"], repeat=repeat,
                             name=model_cfg["name"])
    raise ValueError(kind)


def first_reaching(iters, thr):
    for it in iters:
        if it["ind"]["R1"] >= thr:
            return it["k"]
    return None


def convergence_iter(iters, eps=0.01):
    """T6: pirmoji iteracija, nuo kurios kokybės funkcijos pokytis < eps iki ciklo pabaigos."""
    sc = [it["score"] for it in iters]
    for k in range(1, len(sc)):
        if all(abs(sc[j] - sc[j - 1]) < eps for j in range(k, len(sc))):
            return iters[k]["k"]
    return None


# ------------------------------------------------------------------------------ vienas pakartojimas
def run_repeat(sid: str, repeat: int, a: dict, model_cfg: dict | None) -> dict:
    scn = build_registry()[sid]
    if a.get("overrides"):
        scn.overrides = {**scn.overrides, **a["overrides"]}
    params = params_for_profile(scn.profile, scn.overrides)
    theta0 = dict(DEFAULT_THETA)
    n_iter, m, n_eval = a["iterations"], a["inner"], a["eval"]
    eval_seeds = [sim_seed(a["base_seed"], repeat, 1000 + j) for j in range(n_eval)]
    researcher = build_researcher(scn, repeat, a, model_cfg)
    t0 = time.time()
    iters, history, theta = [], [], dict(theta0)
    tech = dict(requests=0, format_errors=0, bound_violations=0, api_errors=0, prompt_tokens=0,
                completion_tokens=0, llm_time=0.0, just_ok=0, just_n=0, accepted=0, proposals=0)

    if researcher.iterative:
        for k in range(1, n_iter + 1):
            if a["crn_across_iterations"]:
                seeds = [sim_seed(a["base_seed"], repeat, j) for j in range(m)]
            else:
                seeds = [sim_seed(a["base_seed"], repeat, 100 * k + j) for j in range(m)]
            ind = evaluate(params, theta, seeds)
            rec = dict(k=k, theta=dict(theta), ind=ind, score=score(ind))
            history.append(rec)
            prop = researcher.propose(k, n_iter, theta, ind, history, {"m": m})
            rec["accepted"] = prop.accepted
            rec["change"] = prop.meta.get("change", {})
            rec["meta"] = prop.meta
            if scn.researcher == "llm":
                tech["proposals"] += 1
                tech["accepted"] += int(prop.accepted)
                for key in ("requests", "format_errors", "bound_violations", "api_errors", "prompt_tokens",
                            "completion_tokens", "just_ok", "just_n"):
                    tech[key] += prop.meta.get(key, 0)
                tech["llm_time"] += prop.meta.get("latency", 0.0)
            iters.append(rec)
            theta = dict(prop.theta)

    theta_last = dict(theta)
    theta_best = dict(max(iters, key=lambda r: r["score"])["theta"]) if iters else dict(theta0)
    final = {}
    for tag, th in (("last", theta_last), ("best", theta_best)):
        if tag == "best" and th == theta_last:
            final["best"] = final["last"]
            continue
        ind = evaluate(params, th, eval_seeds)
        final[tag] = dict(theta=th, ind=ind, score=score(ind))
    return dict(
        scenario=sid, repeat=repeat, model=(model_cfg or {}).get("name"), profile=scn.profile,
        researcher=scn.researcher, overrides=scn.overrides, args={k: a[k] for k in (
            "base_seed", "iterations", "inner", "eval", "lang", "crn_across_iterations")},
        theta0=theta0, iterations=iters, final=final,
        R9=dict(min=first_reaching(iters, NORM_MIN), target=first_reaching(iters, NORM_TARGET)),
        T6=convergence_iter(iters) if iters else None, tech=tech, time_s=time.time() - t0)


def _job(args):
    sid, repeat, a, model_cfg, outdir = args
    path = Path(outdir) / f"rep_{repeat:04d}.json"
    if path.exists() and not a["force"]:
        return repeat, "skip"
    res = run_repeat(sid, repeat, a, model_cfg)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return repeat, "ok"


# ------------------------------------------------------------------------------ tinklelis (G1/G2)
def _grid_point(args):
    theta, sid, a = args
    scn = build_registry()[sid]
    params = params_for_profile(scn.profile, {**scn.overrides, **(a.get("overrides") or {})})
    seeds = [sim_seed(a["base_seed"], r, 1000 + j) for r in range(a["grid_repeats"]) for j in range(a["eval"])]
    ind = evaluate(params, theta, seeds)
    return dict(**theta, score=score(ind), **{k: ind[k] for k in ("R1", "R4_need", "R5_share", "R8", "R10", "R3_queue")})


def run_grid(sid: str, a: dict, outdir: Path, workers: int):
    import pandas as pd
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_grid_point, [(c, sid, a) for c in combos]))
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "grid.csv", index=False)
    print(df.head(5).to_string(index=False))
    print(f"Geriausia kombinacija įrašyta: {outdir/'grid.csv'}")


# ------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--scenario")
    ap.add_argument("--model", help="DKM pavadinimas iš configs/models.yaml (llm scenarijams; MOCK testavimui)")
    ap.add_argument("--models-file", default="configs/models.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--repeats", type=int, help="pakartojimų skaičius (numatytasis – iš scenarijaus)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--eval", type=int, default=20, help="held-out simuliacijų skaičius galutiniam vertinimui")
    ap.add_argument("--grid-repeats", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=20260)
    ap.add_argument("--lang", default="lt", choices=["lt", "en"])
    ap.add_argument("--memory-window", type=int, default=5)
    ap.add_argument("--max-retries", type=int, default=1, help="DKM pakartojimai po formato klaidos")
    ap.add_argument("--no-crn-iterations", action="store_true", help="skirtingos sėklos kiekvienoje iteracijoje")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="perrašyti SimParams parametrą")
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()

    reg = build_registry()
    if ns.list:
        for s in reg.values():
            print(f"{s.id:10s} profil={s.profile:4s} tyrėjas={s.researcher:9s} pakart.={s.repeats:3d}  {s.desc}")
        return
    if not ns.scenario:
        ap.error("nurodykite --scenario arba --list")
    scn = reg[ns.scenario]
    overrides = {}
    for kv in ns.set:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v) if v[0] in "-0123456789tfn[" else v
    a = dict(base_seed=ns.base_seed, iterations=ns.iterations, inner=ns.inner, eval=ns.eval, lang=ns.lang,
             memory_window=ns.memory_window, max_retries=ns.max_retries, overrides=overrides, force=ns.force,
             crn_across_iterations=not ns.no_crn_iterations, grid_repeats=ns.grid_repeats)

    model_cfg, label = None, scn.id
    if scn.researcher == "llm":
        from .llm_client import load_model_cfg
        name = ns.model or scn.model
        if not name:
            ap.error(f"{scn.id}: nurodykite --model (pvz., geriausią iš S5–S7)")
        model_cfg = load_model_cfg(ns.models_file, name)
        label = f"{scn.id}__{name}"
        if model_cfg.get("provider") != "mock":            # patikra prieš paleidimą: 1 trumpa užklausa
            from .llm_client import make_client
            try:
                probe = make_client(dict(model_cfg, api_retries=2, max_tokens=20)).chat(
                    [{"role": "user", "content": 'Atsakyk tik JSON objektu: {"ok": true}'}])
            except Exception as e:                          # noqa: BLE001
                raise SystemExit(f"[{label}] DKM nepasiekiamas: {e}")
            if probe.error:
                raise SystemExit(f"[{label}] DKM nepasiekiamas, eksperimentas nepradėtas: {probe.error}")
            print(f"[{label}] DKM patikra ok: {model_cfg.get('model')} ({probe.latency:.1f} s)", flush=True)
    outdir = Path(ns.out) / label

    if scn.researcher == "grid":
        run_grid(scn.id, a, outdir, ns.workers)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    n = ns.repeats if ns.repeats is not None else scn.repeats
    jobs = [(scn.id, r, a, model_cfg, str(outdir)) for r in range(ns.start, ns.start + n)]
    Pool = ThreadPoolExecutor if scn.researcher == "llm" else ProcessPoolExecutor   # LLM – I/O; kita – CPU
    t0, done = time.time(), 0
    with Pool(ns.workers) as ex:
        for repeat, status in ex.map(_job, jobs):
            done += 1
            print(f"[{label}] {done}/{len(jobs)} pakartojimas {repeat}: {status}  ({time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()