import numpy as np


ACTION_NO_EDIT = 0
ACTION_EDIT = 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class World:
    def __init__(
        self,
        num_users=200,
        num_videos=200,
        num_brands=5,
        dim=12,
        beta0=0.0,
        gamma=0.5,
        delta=0.01,
        eta=0.35,
        kappa_low=0.8,
        kappa_high=1.2,
        editability_alpha=2.0,
        editability_beta=2.0,
        user_to_cohort=None,
        num_cohorts=1,
        cohort_noise=0.1,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)
        self.num_users = num_users
        self.num_videos = num_videos
        self.num_brands = num_brands
        self.dim = dim
        self.beta0 = beta0
        self.gamma = gamma
        self.delta = delta
        self.eta = eta

        if user_to_cohort is not None:
            if len(user_to_cohort) != num_users:
                raise ValueError("user_to_cohort length must match num_users")
            if num_cohorts < 1:
                raise ValueError("num_cohorts must be >= 1")
            cohort_vecs = self._sample_unit_vectors(num_cohorts, dim)
            cohort_centered = cohort_vecs - cohort_vecs.mean(axis=0, keepdims=True)
            noise = self.rng.normal(size=(num_users, dim))
            self.p_u = cohort_centered[user_to_cohort] + cohort_noise * noise
        else:
            self.p_u = self._sample_unit_vectors(num_users, dim)
        self.x_v = self._sample_unit_vectors(num_videos, dim)
        self.q_b = self._sample_unit_vectors(num_brands, dim)
        self.q_hat = self.q_b

        self.s_v = self.rng.beta(editability_alpha, editability_beta, size=num_videos)
        self.kappa_b = self.rng.uniform(kappa_low, kappa_high, size=num_brands)

    def _sample_unit_vectors(self, n, d):
        x = self.rng.normal(size=(n, d))
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def apply_edit(self, video_id, brand_id):
        return self.x_v[video_id] + self.eta * self.q_hat[brand_id]

    def is_edit_acceptable(self, video_id, brand_id):
        return self.eta <= self.kappa_b[brand_id] * self.s_v[video_id]

    def expected_ctr(self, user_id, video_id, brand_id, action_id):
        p_u = self.p_u[user_id]
        q_b = self.q_b[brand_id]
        if action_id == ACTION_EDIT:
            x_vp = self.apply_edit(video_id, brand_id)
        else:
            x_vp = self.x_v[video_id]
        x_vp_norm = x_vp / (np.linalg.norm(x_vp) + 1e-12)
        logit = (
            self.beta0
            + np.dot(p_u, x_vp_norm)
            + self.gamma * np.dot(p_u, q_b)
            + self.delta * np.dot(q_b, x_vp_norm)
        )
        return float(sigmoid(logit))

    def sample_click(self, user_id, video_id, brand_id, action_id, rng):
        p = self.expected_ctr(user_id, video_id, brand_id, action_id)
        return 1 if rng.random() < p else 0
