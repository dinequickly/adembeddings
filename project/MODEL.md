You are a mathematical reviewer.
I am defining a synthetic world for ad personalization.

Please rewrite my CTR model in a clean, identifiable form and verify:
- all dimensions are consistent
- outputs are in [0,1]
- the model reduces to standard recommendation if no edits are allowed

Model ingredients:
- user vector p_u ∈ R^d
- video vector x_v ∈ R^d
- brand vector q_b ∈ R^d
- edit action produces x_v' = T(x_v, b, a)
- scalar coefficients gamma, delta for brand effects

I want a logit-additive CTR model. Return the final equation + a short justification.

Stop here. Do not proceed until the equation is clean.

Final equation:
CTR_{u,v,b,a} = σ( β0 + p_u^T x_v' + gamma * p_u^T q_b + delta * q_b^T x_v' ) ,
with x_v' = T(x_v, b, a) and σ(z)=1/(1+e^{-z}).

Justification:
- Dimensions: p_u,x_v',q_b ∈ R^d so each inner product is a scalar; β0, gamma, delta are scalars; logit is scalar.
- Range: σ maps any real logit to (0,1), so CTR ∈ (0,1) (can be treated as [0,1] with limits).
- Brand effects: p_u^T q_b captures user-brand preference and q_b^T x_v' captures brand-video fit, so brands can differ even for the same unedited video.
- No-edit reduction: if edits are not allowed, x_v' = x_v, giving CTR = σ(β0 + p_u^T x_v + gamma * p_u^T q_b + delta * q_b^T x_v). Setting gamma=delta=0 recovers standard MF.

---

You are a bandit/decision-theory expert.

Given this simulation, define:
- what an “arm” is
- what the reward is
- what information is observed
- how Thompson sampling or UCB applies
- what baselines to compare against

Output should be math + pseudocode.

Definitions (math):
- Context at round t: c_t = (u_t, V_t, B_t), where u_t is the user, V_t is a candidate video set, B_t is a candidate brand set.
- Action/arm: a_t = (v_t, b_t, action_id) with v_t ∈ V_t, b_t ∈ B_t, action_id ∈ {no-edit, edit}; must satisfy acceptability Acc(x_v, T(x_v, b_t)) = 1 for edit.
- Edited video: x_{v_t}' = T(x_{v_t}, b_t) if action_id=edit, else x_{v_t}' = x_{v_t}.
- Reward: r_t = y_t ∈ {0,1}, where y_t ~ Bernoulli(σ(β0 + p_{u_t}^T x_{v_t}' + gamma * p_{u_t}^T q_{b_t} + delta * q_{b_t}^T x_{v_t}')).
- Observation: (u_t, v_t, b_t, action_id, r_t).

Thompson sampling (Beta-Bernoulli, discrete arms):
- Maintain per-arm posterior Beta(alpha_{v,b,a}, beta_{v,b,a}) over mean click rate mu_{v,b,a} for arms (v,b,action_id).
- At round t, sample mu~_{v,b,a} ~ Beta(alpha_{v,b,a}, beta_{v,b,a}) for each feasible arm.
- Choose arm_t = argmax mu~_{v,b,a} among feasible arms.
- Observe r_t and update: alpha += r_t, beta += (1 - r_t).

UCB (Bernoulli, discrete arms):
- Maintain per-arm empirical mean mu_hat_{v,b,a} and count n_{v,b,a}.
- Choose arm_t = argmax_{feasible arms} [ mu_hat_{v,b,a} + sqrt(2 * log t / n_{v,b,a}) ].
- Observe r_t and update mu_hat and n.

Baselines:
- No-edit baseline: restrict action_id=no-edit. This recovers standard recommendation when gamma=delta=0.
- Random-edit baseline: choose among feasible arms uniformly at random.
- Brand-only baseline: rank by q_b alignment, ignoring user p_u.
- Greedy baseline: choose the current best arm by estimated CTR without exploration.

Pseudocode (high-level):
1) Initialize model/posterior or CTR estimates.
2) For t = 1..T:
   a) Observe context c_t (user u_t and candidate sets).
   b) Enumerate feasible arms (v,b,action_id) with Acc=1 for edit actions.
   c) If Thompson sampling: sample mu~ per arm and pick argmax.
      If UCB: pick argmax (estimate + bonus).
   d) Execute chosen arm, observe reward r_t.
   e) Update model/posterior with (c_t, a_t, r_t).
