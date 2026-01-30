import numpy as np

from .world import ACTION_NO_EDIT, ACTION_EDIT


def enumerate_feasible_arms(world, candidate_videos, candidate_brands):
    arms = []
    for v in candidate_videos:
        for b in candidate_brands:
            arms.append((v, b, ACTION_NO_EDIT))
            if world.is_edit_acceptable(v, b):
                arms.append((v, b, ACTION_EDIT))
    return arms


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select_arm(self, world, user_id, candidate_videos, candidate_brands, cohort_id=None):
        arms = enumerate_feasible_arms(world, candidate_videos, candidate_brands)
        return arms[self.rng.integers(len(arms))]

    def update(self, arm, successes, failures, cohort_id=None):
        return None


class NoEditGreedyPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select_arm(self, world, user_id, candidate_videos, candidate_brands, cohort_id=None):
        best = None
        best_ctr = -1.0
        for v in candidate_videos:
            for b in candidate_brands:
                ctr = world.expected_ctr(user_id, v, b, ACTION_NO_EDIT)
                if ctr > best_ctr:
                    best_ctr = ctr
                    best = (v, b, ACTION_NO_EDIT)
        return best

    def update(self, arm, successes, failures, cohort_id=None):
        return None


class ThompsonPolicy:
    def __init__(self, num_videos, num_brands, seed=0, alpha0=1.0, beta0=1.0):
        self.rng = np.random.default_rng(seed)
        self.alpha = np.full((num_videos, num_brands, 2), alpha0, dtype=float)
        self.beta = np.full((num_videos, num_brands, 2), beta0, dtype=float)

    def select_arm(self, world, user_id, candidate_videos, candidate_brands, cohort_id=None):
        best = None
        best_sample = -1.0
        for v in candidate_videos:
            for b in candidate_brands:
                a0 = self.alpha[v, b, ACTION_NO_EDIT]
                b0 = self.beta[v, b, ACTION_NO_EDIT]
                mu0 = self.rng.beta(a0, b0)
                if mu0 > best_sample:
                    best_sample = mu0
                    best = (v, b, ACTION_NO_EDIT)
                if world.is_edit_acceptable(v, b):
                    a1 = self.alpha[v, b, ACTION_EDIT]
                    b1 = self.beta[v, b, ACTION_EDIT]
                    mu1 = self.rng.beta(a1, b1)
                    if mu1 > best_sample:
                        best_sample = mu1
                        best = (v, b, ACTION_EDIT)
        return best

    def update(self, arm, successes, failures, cohort_id=None):
        v, b, a = arm
        self.alpha[v, b, a] += successes
        self.beta[v, b, a] += failures


class CohortThompsonPolicy:
    def __init__(self, num_cohorts, num_videos, num_brands, seed=0, alpha0=1.0, beta0=1.0):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31 - 1, size=num_cohorts)
        self.policies = [
            ThompsonPolicy(num_videos, num_brands, seed=int(s), alpha0=alpha0, beta0=beta0)
            for s in seeds
        ]

    def select_arm(self, world, user_id, candidate_videos, candidate_brands, cohort_id=None):
        if cohort_id is None:
            raise ValueError("cohort_id required for CohortThompsonPolicy")
        return self.policies[cohort_id].select_arm(
            world, user_id, candidate_videos, candidate_brands, cohort_id=cohort_id
        )

    def update(self, arm, successes, failures, cohort_id=None):
        if cohort_id is None:
            raise ValueError("cohort_id required for CohortThompsonPolicy")
        self.policies[cohort_id].update(arm, successes, failures, cohort_id=cohort_id)


class OraclePolicy:
    def select_arm(self, world, user_id, candidate_videos, candidate_brands, cohort_id=None):
        best = None
        best_ctr = -1.0
        for v in candidate_videos:
            for b in candidate_brands:
                ctr0 = world.expected_ctr(user_id, v, b, ACTION_NO_EDIT)
                if ctr0 > best_ctr:
                    best_ctr = ctr0
                    best = (v, b, ACTION_NO_EDIT)
                if world.is_edit_acceptable(v, b):
                    ctr1 = world.expected_ctr(user_id, v, b, ACTION_EDIT)
                    if ctr1 > best_ctr:
                        best_ctr = ctr1
                        best = (v, b, ACTION_EDIT)
        return best

    def update(self, arm, successes, failures, cohort_id=None):
        return None
