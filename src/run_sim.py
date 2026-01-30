import argparse
import os

import numpy as np

from .world import World, ACTION_EDIT, ACTION_NO_EDIT
from .policies import (
    RandomPolicy,
    NoEditGreedyPolicy,
    ThompsonPolicy,
    CohortThompsonPolicy,
    OraclePolicy,
)


def make_contexts(
    world,
    num_rounds,
    candidate_videos,
    candidate_brands,
    user_to_cohort,
    segment_len=1,
    seed=0,
):
    rng = np.random.default_rng(seed)
    contexts = []
    current_user = None
    for _ in range(num_rounds):
        if current_user is None or (segment_len > 1 and len(contexts) % segment_len == 0):
            current_user = int(rng.integers(world.num_users))
        u = current_user
        vids = rng.choice(world.num_videos, size=candidate_videos, replace=False)
        if candidate_brands >= world.num_brands:
            brands = np.arange(world.num_brands, dtype=int)
        else:
            brands = rng.choice(world.num_brands, size=candidate_brands, replace=False)
        cohort_id = int(user_to_cohort[u])
        contexts.append((u, cohort_id, vids, brands))
    return contexts


def compute_acceptability_stats(world, contexts):
    total = 0
    rejected = 0
    better_total = 0
    better = 0
    for u, _cohort, vids, brands in contexts:
        for v in vids:
            for b in brands:
                total += 1
                if not world.is_edit_acceptable(v, b):
                    rejected += 1
                    continue
                better_total += 1
                ctr_edit = world.expected_ctr(u, v, b, ACTION_EDIT)
                ctr_no = world.expected_ctr(u, v, b, ACTION_NO_EDIT)
                if ctr_edit > ctr_no:
                    better += 1
    rejected_frac = rejected / total if total else 0.0
    better_frac = better / better_total if better_total else 0.0
    return rejected_frac, better_frac


def simulate_policy(world, policy, contexts, seed=0, impressions_per_pull=1):
    rng = np.random.default_rng(seed)
    successes = np.zeros(len(contexts), dtype=float)
    for t, (u, cohort_id, vids, brands) in enumerate(contexts):
        arm = policy.select_arm(world, u, vids, brands, cohort_id=cohort_id)
        v, b, a = arm
        p = world.expected_ctr(u, v, b, a)
        succ = rng.binomial(impressions_per_pull, p)
        fail = impressions_per_pull - succ
        policy.update(arm, succ, fail, cohort_id=cohort_id)
        successes[t] = succ
    return successes


def cumulative_rate(successes, impressions_per_pull):
    denom = (np.arange(len(successes)) + 1) * impressions_per_pull
    return np.cumsum(successes) / denom


def save_plot(out_path, series_dict):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, series in series_dict.items():
        plt.plot(series, label=name)
    plt.xlabel("Round")
    plt.ylabel("Average click rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--candidate-videos", type=int, default=12)
    parser.add_argument("--candidate-brands", type=int, default=5)
    parser.add_argument("--num-cohorts", type=int, default=1)
    parser.add_argument("--segment-len", type=int, default=1)
    parser.add_argument("--impressions-per-pull", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default="click_rate.png")
    args = parser.parse_args()

    if args.num_cohorts < 1:
        raise ValueError("num_cohorts must be >= 1")
    if args.segment_len < 1:
        raise ValueError("segment_len must be >= 1")
    if args.impressions_per_pull < 1:
        raise ValueError("impressions_per_pull must be >= 1")

    num_users = 200
    rng = np.random.default_rng(args.seed + 99)
    user_to_cohort = rng.integers(0, args.num_cohorts, size=num_users)
    world = World(
        seed=args.seed,
        num_users=num_users,
        user_to_cohort=user_to_cohort,
        num_cohorts=args.num_cohorts,
    )

    contexts = make_contexts(
        world,
        args.rounds,
        args.candidate_videos,
        args.candidate_brands,
        user_to_cohort,
        segment_len=args.segment_len,
        seed=args.seed + 1,
    )

    rejected_frac, better_frac = compute_acceptability_stats(world, contexts)
    print(f"Rejected edited-arm fraction: {rejected_frac:.3f}")
    print(f"Edited-better-than-no-edit fraction (feasible edits): {better_frac:.3f}")

    if args.num_cohorts > 1:
        ts_policy = CohortThompsonPolicy(
            args.num_cohorts, world.num_videos, world.num_brands, seed=args.seed + 4
        )
    else:
        ts_policy = ThompsonPolicy(world.num_videos, world.num_brands, seed=args.seed + 4)

    policies = {
        "random": RandomPolicy(seed=args.seed + 2),
        "no_edit_greedy": NoEditGreedyPolicy(seed=args.seed + 3),
        "thompson": ts_policy,
        "oracle_constrained": OraclePolicy(),
    }

    results = {}
    for name, policy in policies.items():
        succ = simulate_policy(
            world, policy, contexts, seed=args.seed + 10, impressions_per_pull=args.impressions_per_pull
        )
        results[name] = succ

    denom = args.rounds * args.impressions_per_pull
    print(f"random policy average click rate: {results['random'].sum() / denom:.3f}")
    print(f"no-edit greedy policy average click rate: {results['no_edit_greedy'].sum() / denom:.3f}")
    print(f"Thompson sampling policy average click rate: {results['thompson'].sum() / denom:.3f}")
    print(f"constrained oracle average click rate: {results['oracle_constrained'].sum() / denom:.3f}")

    series = {
        "random": cumulative_rate(results["random"], args.impressions_per_pull),
        "no-edit greedy": cumulative_rate(results["no_edit_greedy"], args.impressions_per_pull),
        "Thompson": cumulative_rate(results["thompson"], args.impressions_per_pull),
        "Oracle (constrained)": cumulative_rate(
            results["oracle_constrained"], args.impressions_per_pull
        ),
    }
    out_path = os.path.abspath(args.plot)
    save_plot(out_path, series)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
