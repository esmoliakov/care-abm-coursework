"""Rezultatų analizė (4 skyriui): lentelės (CSV), statistiniai testai, paveikslai.

  python -m care_abm.analysis.analyze --results results --out analysis_out
  python -m care_abm.analysis.analyze --pairs "S5__DKM-A:S4,S5__DKM-A:S3" --policy last

Kiekvienam rodikliui: vidurkis, mediana, 95 % bootstrap PI. Sulyginamiems scenarijams (tos pačios held-out
sėklos pagal pakartojimo nr.) – Wilcoxon pasirašytų rangų testas + rangų-bineris efektas, Holm korekcija;
nepriklausomoms imtims – Mann–Whitney (+ Cliff delta); daugiau nei 2 grupės – Kruskal–Wallis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

IND = ["score", "R1", "R1b", "R2", "R3_queue", "R3_wait", "R4", "R4_need", "R5_B", "R5_share", "R6", "R7",
       "R8", "R10", "R11"]


# --------------------------------------------------------------------------------------- įkėlimas
def load(results_dir: str, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, iters = [], []
    for f in sorted(Path(results_dir).glob("*/rep_*.json")):
        raw = f.read_bytes()
        try:
            r = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:                 # seni Windows (cp1257) failai
            r = json.loads(raw.decode("cp1257"))
        label = f.parent.name
        fin = r["final"][policy]
        row = dict(label=label, repeat=r["repeat"], profile=r["profile"], researcher=r["researcher"],
                   shocked=bool(r["overrides"]), score=fin["score"], **{k: v for k, v in fin["ind"].items()},
                   R9_min=r["R9"]["min"], R9_target=r["R9"]["target"], T6=r["T6"], time_s=r["time_s"],
                   **{f"tech_{k}": v for k, v in r["tech"].items()})
        for k, v in fin["theta"].items():
            row[f"theta_{k}"] = v
        rows.append(row)
        for it in r["iterations"]:
            iters.append(dict(label=label, repeat=r["repeat"], k=it["k"], score=it["score"], R1=it["ind"]["R1"],
                              R4_need=it["ind"]["R4_need"]))
    return pd.DataFrame(rows), pd.DataFrame(iters)


# --------------------------------------------------------------------------------------- statistika
def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial_paired(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def cliffs_delta(x, y):
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def compare_pairs(df, pairs, metrics=("score", "R1", "R4_need")):
    out = []
    for a, b in pairs:
        A, B = df[df.label == a].set_index("repeat"), df[df.label == b].set_index("repeat")
        common = A.index.intersection(B.index)
        for met in metrics:
            if len(common) >= 6:                                    # suporuotos (ta pati sėkla) imtys
                x, y = A.loc[common, met].to_numpy(), B.loc[common, met].to_numpy()
                d = x - y
                try:
                    p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    p = 1.0
                out.append(dict(A=a, B=b, metric=met, test="Wilcoxon (suporuota)", n=len(common),
                                mean_A=x.mean(), mean_B=y.mean(), diff=d.mean(), p=p,
                                effect=rank_biserial_paired(d), effect_name="rangų-bineris"))
            else:
                x, y = A[met].dropna().to_numpy(), B[met].dropna().to_numpy()
                if len(x) < 3 or len(y) < 3:
                    continue
                p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                out.append(dict(A=a, B=b, metric=met, test="Mann–Whitney", n=f"{len(x)}/{len(y)}",
                                mean_A=x.mean(), mean_B=y.mean(), diff=x.mean() - y.mean(), p=p,
                                effect=cliffs_delta(x, y), effect_name="Cliff delta"))
    res = pd.DataFrame(out)
    if not res.empty:
        res["p_holm"] = np.nan
        for met, idx in res.groupby("metric").groups.items():          # Holm kiekvienam rodikliui atskirai
            res.loc[idx, "p_holm"] = holm(res.loc[idx, "p"].to_numpy())
    return res


def kruskal(df, labels, metrics=("score", "R1", "R4_need")):
    out = []
    for met in metrics:
        groups = [df[df.label == l][met].dropna().to_numpy() for l in labels if (df.label == l).any()]
        if len(groups) >= 3 and all(len(g) >= 3 for g in groups):
            h, p = stats.kruskal(*groups)
            out.append(dict(labels=",".join(labels), metric=met, H=h, p=p))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------- lentelės
def summary_table(df):
    rows = []
    for label, g in df.groupby("label"):
        for m in IND:
            if m not in g:
                continue
            lo, hi = boot_ci(g[m])
            rows.append(dict(label=label, indicator=m, n=int(g[m].notna().sum()), mean=g[m].mean(),
                             median=g[m].median(), ci95_lo=lo, ci95_hi=hi))
    return pd.DataFrame(rows)


def technical_table(df):
    d = df[df.researcher == "llm"]
    rows = []
    for label, g in d.groupby("label"):
        req = g["tech_requests"].sum()
        rows.append(dict(
            label=label, runs=len(g),
            T1_prompt_tokens_per_run=g["tech_prompt_tokens"].mean(),
            T1_completion_tokens_per_run=g["tech_completion_tokens"].mean(),
            T2_requests_per_run=g["tech_requests"].mean(),
            T3_format_error_share=g["tech_format_errors"].sum() / max(req, 1),
            T4_bound_violation_share=g["tech_bound_violations"].sum() / max(g["tech_proposals"].sum(), 1),
            T4b_api_error_share=g["tech_api_errors"].sum() / max(req, 1),
            T5_llm_s_per_request=g["tech_llm_time"].sum() / max(req, 1),
            T5_run_time_s=g["time_s"].mean(),
            T6_convergence_iter_median=g["T6"].median(),
            T7_justification_accuracy=g["tech_just_ok"].sum() / max(g["tech_just_n"].sum(), 1),
            accepted_share=g["tech_accepted"].sum() / max(g["tech_proposals"].sum(), 1),
            R9_min_reached_share=g["R9_min"].notna().mean(), R9_min_median=g["R9_min"].median(),
            R9_target_reached_share=g["R9_target"].notna().mean(), R9_target_median=g["R9_target"].median()))
    return pd.DataFrame(rows)


def regret_table(df, results_dir):
    rows = []
    for prof, gid in (("low", "G1"), ("high", "G2")):
        f = Path(results_dir) / gid / "grid.csv"
        if not f.exists():
            continue
        grid = pd.read_csv(f)
        best = grid.sort_values("score", ascending=False).iloc[0]
        for label, g in df[(df.profile == prof) & (~df.shocked)].groupby("label"):
            rows.append(dict(label=label, profile=prof, grid_best_score=best["score"],
                             grid_best_theta=f"a={best.alpha_amb}, L={best.l_max}, r={best.respite}",
                             mean_final_score=g["score"].mean(), mean_regret=best["score"] - g["score"].mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- paveikslai
def figures(df, iters, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not iters.empty:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for label, g in iters.groupby("label"):
            for a_, met, title in ((ax[0], "R1", "R1 (gavėjai / 1000 senjorų)"), (ax[1], "score", "Kokybės funkcija")):
                by = g.groupby("k")[met]
                m = by.mean()
                ci = by.apply(lambda s: boot_ci(s, 500))
                a_.plot(m.index, m.values, marker="o", ms=3, label=label)
                a_.fill_between(m.index, [c[0] for c in ci], [c[1] for c in ci], alpha=0.15)
                a_.set_xlabel("Iteracija")
                a_.set_title(title)
        ax[0].axhline(61, ls="--", c="red", lw=0.8)
        ax[0].axhline(107, ls="--", c="blue", lw=0.8)
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_trajektorijos.png", dpi=200)
        plt.close(fig)

    labels = sorted(df.label.unique())
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a_, met, title in zip(ax, ("score", "R1", "R4_need"),
                              ("Kokybės funkcija (galutinė)", "R1 (galutinis)", "Globos skurdas, % turinčių poreikį")):
        a_.boxplot([df[df.label == l][met].dropna() for l in labels], showfliers=False)
        a_.set_xticks(range(1, len(labels) + 1))
        a_.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        a_.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "fig_galutiniai.png", dpi=200)
    plt.close(fig)

    t = technical_table(df)
    if not t.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["T1_prompt_tokens_per_run"], 0.4, label="įvestis")
        ax.bar(x + 0.2, t["T1_completion_tokens_per_run"], 0.4, label="išvestis")
        ax.set_xticks(x)
        ax.set_xticklabels(t["label"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("žetonai / paleidimas")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig_zetonai.png", dpi=200)
        plt.close(fig)


# --------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--policy", default="last", choices=["last", "best"],
                    help="last – galutinė tyrėjo rekomendacija; best – geriausia ciklo metu")
    ap.add_argument("--pairs", default="", help="pvz. S5__DKM-A:S4,S5__DKM-A:S3 (A:B => A palyginama su B)")
    ap.add_argument("--kruskal", default="", help="kableliais atskirti scenarijai Kruskal–Wallis testui")
    ns = ap.parse_args()

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    df, iters = load(ns.results, ns.policy)
    if df.empty:
        raise SystemExit("Rezultatų nerasta.")
    labels = sorted(df.label.unique())

    if ns.pairs:
        pairs = [tuple(p.split(":")) for p in ns.pairs.split(",")]
    else:                                                       # automatiniai palyginimai
        pairs = []
        for l in labels:
            if l.startswith(("S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14")):
                pairs += [(l, b) for b in labels if b in ("S3", "S4", "S4b")]
            if l in ("S3", "S4", "S4b"):
                pairs += [(l, "S1")] if "S1" in labels else []
    pairs = [(a, b) for a, b in pairs if a in labels and b in labels]
    kr = ns.kruskal.split(",") if ns.kruskal else [l for l in labels if l.startswith(("S5__", "S6__", "S7__"))]

    summary_table(df).to_csv(out / "suvestine.csv", index=False)
    technical_table(df).to_csv(out / "techniniai_rodikliai.csv", index=False)
    regret_table(df, ns.results).to_csv(out / "regret_vs_tinklelis.csv", index=False)
    cmp_ = compare_pairs(df, pairs)
    cmp_.to_csv(out / "testai_poros.csv", index=False)
    kruskal(df, kr).to_csv(out / "testai_kruskal.csv", index=False)
    df.to_csv(out / "visi_paleidimai.csv", index=False)
    figures(df, iters, out)

    pd.set_option("display.width", 200)
    print(f"Politika: {ns.policy}. Scenarijai: {', '.join(labels)}")
    s = summary_table(df)
    print(s[s.indicator.isin(["score", "R1", "R4_need", "R5_share"])].round(3).to_string(index=False))
    if not cmp_.empty:
        print(cmp_.round(4).to_string(index=False))
    print("Išsaugota:", out.resolve())


if __name__ == "__main__":
    main()