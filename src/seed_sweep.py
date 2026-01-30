import argparse
import csv
import os

import numpy as np

from .world import World
from .run_sim import make_contexts, compute_acceptability_stats, simulate_policy
from .policies import (
    RandomPolicy,
    NoEditGreedyPolicy,
    ThompsonPolicy,
    CohortThompsonPolicy,
    OraclePolicy,
)


def run_once(
    seed,
    rounds,
    candidate_videos,
    candidate_brands,
    num_cohorts,
    segment_len,
    impressions_per_pull,
):
    num_users = 200
    rng = np.random.default_rng(seed + 99)
    user_to_cohort = rng.integers(0, num_cohorts, size=num_users)
    world = World(
        seed=seed,
        num_users=num_users,
        user_to_cohort=user_to_cohort,
        num_cohorts=num_cohorts,
    )
    contexts = make_contexts(
        world,
        rounds,
        candidate_videos,
        candidate_brands,
        user_to_cohort,
        segment_len=segment_len,
        seed=seed + 1,
    )

    rejected_frac, better_frac = compute_acceptability_stats(world, contexts)

    if num_cohorts > 1:
        ts_policy = CohortThompsonPolicy(
            num_cohorts, world.num_videos, world.num_brands, seed=seed + 4
        )
    else:
        ts_policy = ThompsonPolicy(world.num_videos, world.num_brands, seed=seed + 4)

    policies = {
        "random": RandomPolicy(seed=seed + 2),
        "no_edit_greedy": NoEditGreedyPolicy(seed=seed + 3),
        "thompson": ts_policy,
        "oracle_constrained": OraclePolicy(),
    }

    results = {}
    for name, policy in policies.items():
        succ = simulate_policy(
            world,
            policy,
            contexts,
            seed=seed + 10,
            impressions_per_pull=impressions_per_pull,
        )
        results[name] = succ.sum() / (rounds * impressions_per_pull)

    return {
        "rejected_frac": rejected_frac,
        "better_frac": better_frac,
        **results,
    }


def summarize(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return vals.mean(), vals.std(ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--candidate-videos", type=int, default=12)
    parser.add_argument("--candidate-brands", type=int, default=5)
    parser.add_argument("--num-cohorts", type=int, default=8)
    parser.add_argument("--segment-len", type=int, default=1)
    parser.add_argument("--impressions-per-pull", type=int, default=10)
    parser.add_argument("--out-csv", type=str, default="results/seed_sweep.csv")
    parser.add_argument("--out-summary", type=str, default="results/summary_table.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    seed_list = list(range(args.seeds))
    variants = [
        ("main", args.num_cohorts, args.segment_len),
        ("ablation_no_cohorts", 1, args.segment_len),
    ]

    all_rows = []
    for variant, num_cohorts, segment_len in variants:
        for seed in seed_list:
            metrics = run_once(
                seed=seed,
                rounds=args.rounds,
                candidate_videos=args.candidate_videos,
                candidate_brands=args.candidate_brands,
                num_cohorts=num_cohorts,
                segment_len=segment_len,
                impressions_per_pull=args.impressions_per_pull,
            )
            row = {
                "variant": variant,
                "seed": seed,
                "rounds": args.rounds,
                "candidate_videos": args.candidate_videos,
                "candidate_brands": args.candidate_brands,
                "num_cohorts": num_cohorts,
                "segment_len": segment_len,
                "impressions_per_pull": args.impressions_per_pull,
                **metrics,
            }
            all_rows.append(row)

    fieldnames = [
        "variant",
        "seed",
        "rounds",
        "candidate_videos",
        "candidate_brands",
        "num_cohorts",
        "segment_len",
        "impressions_per_pull",
        "rejected_frac",
        "better_frac",
        "random",
        "no_edit_greedy",
        "thompson",
        "oracle_constrained",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_lines = []
    summary_lines.append("| Variant | Random (mean±std) | No-edit (mean±std) | Thompson (mean±std) | Oracle (mean±std) | TS - Random | TS - No-edit |")
    summary_lines.append("|---|---|---|---|---|---|---|")

    for variant, _num_cohorts, _segment_len in variants:
        rows = [r for r in all_rows if r["variant"] == variant]
        rand_mean, rand_std = summarize(rows, "random")
        ne_mean, ne_std = summarize(rows, "no_edit_greedy")
        ts_mean, ts_std = summarize(rows, "thompson")
        or_mean, or_std = summarize(rows, "oracle_constrained")
        uplift_rand = ts_mean - rand_mean
        uplift_ne = ts_mean - ne_mean
        summary_lines.append(
            f"| {variant} | {rand_mean:.3f}±{rand_std:.3f} | {ne_mean:.3f}±{ne_std:.3f} | "
            f"{ts_mean:.3f}±{ts_std:.3f} | {or_mean:.3f}±{or_std:.3f} | "
            f"{uplift_rand:.3f} | {uplift_ne:.3f} |"
        )

    with open(args.out_summary, "w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
