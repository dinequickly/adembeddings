You are designing a simulation to test a recommender + editing system.

Define:
- how users, videos, brands are sampled
- how edits change videos
- how acceptability is enforced
- how clicks are generated

Requirements:
- minimal but expressive
- supports bandit learning
- supports constrained optimization

Output must be a step-by-step algorithm description, not code.

Algorithm (step-by-step, no code):
1) Set global dimensions and priors.
   - Choose embedding dimension d, number of users |U|, videos |V|, brands |B|.
   - Fix global scalars: CTR intercept beta0, brand coefficients gamma and delta, edit strength eta, and per-brand acceptability threshold kappa_b.
   - Define distributions for latent vectors (e.g., isotropic Gaussian) for users, videos, and brands.
   - Tuning guidance: eta should move CTR but be small enough to sometimes fail acceptability; kappa_b should reject a noticeable fraction of edited proposals (e.g., 10-40%).

2) Sample latent entities.
   - For each user u, sample preference vector p_u ∈ R^d.
   - For each video v, sample latent semantics x_v ∈ R^d.
   - For each video v, sample an editability scalar s_v in (0,1) (e.g., Beta distribution).
   - For each brand b, sample latent identity q_b ∈ R^d.

3) Define the edit operator (single learnable action).
   - Use one edit action with a fixed scalar eta.
   - Let q_hat_b = q_b / ||q_b||.
   - Define T(x_v, b) as:
     x_v' = T(x_v, b) = x_v + eta * q_hat_b.
   - Optionally include a no-edit action_id that leaves x_v' = x_v.

4) Enforce acceptability.
   - Define an acceptability function Acc(x_v, x_v') that measures deviation:
     Acc = 1 if ||x_v' − x_v|| <= kappa_b * s_v, else 0.
   - Feasible edits are those with Acc = 1. Infeasible edits are rejected or clipped to the nearest acceptable edit.

5) Generate click probabilities (CTR).
   - For a chosen (u, v, b, action_id), compute edited video:
     if action_id = edit, x_v' = T(x_v, b); if action_id = no-edit, x_v' = x_v.
   - Define logit-additive CTR: logit = beta0 + p_u^T x_v' + gamma * p_u^T q_b + delta * q_b^T x_v'.
   - Convert to probability: CTR = σ(logit).

6) Define arm granularity.
   - Arm = (video_id, brand_id, action_id). (No cohort_id for now.)

7) Add oracle reference policies (for evaluation).
   - Oracle unconstrained: pick the (v, b, action_id) that maximizes CTR over all candidates, ignoring acceptability.
   - Oracle constrained: pick the (v, b, action_id) that maximizes CTR over acceptable candidates only.

8) Learnability check (ground truth sanity check).
   - Verify that editing improves expected CTR for a nontrivial fraction of candidate arms under the constraint (e.g., ~20-60%),
     but not for all users or all arms. If not, resample or retune eta/kappa_b.

9) Sample interactions for the simulator.
   - For each round t, sample a user u (e.g., uniformly or from a demand distribution).
   - Provide candidate sets of videos and brands.
   - The agent selects an action (v, b, action_id) subject to acceptability.
   - Generate click y ~ Bernoulli(CTR).

10) Record observations for learning.
   - Log the context (u, v, b), chosen action_id, acceptability outcome, edited vector x_v', and click y.
   - These logs support bandit learning and constrained optimization (acceptability as a hard constraint).
