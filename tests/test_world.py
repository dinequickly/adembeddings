import numpy as np

from src.world import World, ACTION_EDIT, ACTION_NO_EDIT
from src.run_sim import make_contexts, simulate_policy
from src.policies import RandomPolicy, CohortThompsonPolicy


def test_shapes_and_ctr_range():
    world = World(seed=1, num_users=10, num_videos=12, num_brands=3, dim=8)
    assert world.p_u.shape == (10, 8)
    assert world.x_v.shape == (12, 8)
    assert world.q_b.shape == (3, 8)

    ctr_no = world.expected_ctr(0, 0, 0, ACTION_NO_EDIT)
    ctr_edit = world.expected_ctr(0, 0, 0, ACTION_EDIT)
    assert 0.0 <= ctr_no <= 1.0
    assert 0.0 <= ctr_edit <= 1.0


def test_acceptability_variability():
    world = World(seed=2, num_videos=60, num_brands=2)
    accepts = [world.is_edit_acceptable(v, 0) for v in range(world.num_videos)]
    frac = np.mean(accepts)
    assert 0.1 < frac < 0.9


def test_thompson_beats_random():
    num_users = 200
    num_cohorts = 4
    rng = np.random.default_rng(3)
    user_to_cohort = rng.integers(0, num_cohorts, size=num_users)
    world = World(
        seed=3, num_users=num_users, num_videos=120, num_brands=4, user_to_cohort=user_to_cohort, num_cohorts=num_cohorts
    )
    contexts = make_contexts(
        world,
        num_rounds=2000,
        candidate_videos=12,
        candidate_brands=4,
        user_to_cohort=user_to_cohort,
        seed=4,
    )

    rand_clicks = simulate_policy(world, RandomPolicy(seed=5), contexts, seed=6, impressions_per_pull=5)
    ts_clicks = simulate_policy(
        world,
        CohortThompsonPolicy(num_cohorts, world.num_videos, world.num_brands, seed=7),
        contexts,
        seed=8,
        impressions_per_pull=5,
    )

    assert ts_clicks.mean() >= rand_clicks.mean() + 0.02
