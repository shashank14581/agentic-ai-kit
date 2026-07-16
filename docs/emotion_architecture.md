# Emotion Architecture

**Project:** `agentic-ai-kit`  
**Status:** Pre-implementation specification  
**Architecture version:** 0.1  
**Evidence basis:** Identity-Conditioned Forgetting in Episodic Agent Memory, Version 2

---

## 1. Evidence Boundary

This architecture defines functional, operational affective states for an agent.

It does not claim:

- subjective emotion;
- consciousness;
- sentience;
- human psychological equivalence;
- that operational wound or trauma states imply felt suffering.

The terms used by the mechanism have the following neutral equivalents:

| Research term | Deployment term |
|---|---|
| Success | Confirmed positive outcome |
| Failure | Confirmed negative outcome |
| Wound | Severe corrective failure |
| Trauma | Recurrent or extreme severe failure |

Every state must be derived from observable evidence, identity-conditioned appraisal, and registered transition rules.

No language model may assign wound or trauma solely from free-form interpretation.

---

## 2. System Model

The architecture follows this transition:

$$
I_t,E_t,O_t,H_t
\longrightarrow
A_t
\longrightarrow
Z_t
\longrightarrow
I_{t+1},\Pi_{t+1},M_{t+1}
$$

where:

- $I_t$ is the identity state before the event.
- $E_t$ is the event and action context.
- $O_t$ is observable outcome evidence.
- $H_t$ is relevant episodic history.
- $A_t$ is the appraisal vector.
- $Z_t$ is the continuous functional affective state.
- $I_{t+1}$ is the updated identity.
- $\Pi_{t+1}$ is the updated action policy.
- $M_{t+1}$ is the updated episodic memory.

The architecture maintains three distinct forms of state:

1. **Identity state** — what kind of agent it currently models itself as.
2. **Policy state** — which actions should be repeated or avoided.
3. **Episodic state** — what events remain available for future retrieval.

An event may update policy without updating identity.

A memory may receive high retention without becoming relevant to a future query.

---

## 3. Memory Topology

### 3.1 Logical topology

Episodic memory is represented as a linked sequence of timestep nodes:

$$
M_1 \rightarrow M_2 \rightarrow M_3 \rightarrow \cdots \rightarrow M_t
$$

Each timestep node $M_t$ is the root of an event tree:

```text
M_t
├── context
│   ├── task
│   ├── environment state
│   ├── active identity
│   └── trajectory identifier
├── action
│   ├── action identifier
│   ├── arguments
│   ├── rationale
│   └── alternatives considered
├── expectation
│   ├── predicted outcome
│   ├── confidence
│   └── prediction provenance
├── observation
│   ├── environment response
│   ├── tool response
│   └── state change
├── evidence
│   ├── success evidence
│   ├── failure evidence
│   ├── severity evidence
│   ├── recurrence evidence
│   └── provenance
└── snapshot
    ├── appraisal vector
    ├── affective state
    ├── operational label
    ├── identity effect
    ├── policy effect
    ├── retention properties
    └── configuration version
```

The linked list defines temporal order. The event tree preserves the internal causal and evidential structure of each timestep.

### 3.2 Physical storage

The linked list is a logical model. A production implementation may use:

- relational storage;
- document storage;
- graph storage;
- vector indices;
- append-only event logs;
- compressed archival storage.

The physical backend must preserve:

- timestep order;
- trajectory identifiers;
- parent-child tree edges;
- provenance;
- correction history;
- deletion state;
- model and configuration versions.

Raw evidence should be append-only. Derived appraisals and labels should be versioned and recomputable.

---

## 4. Event and Outcome Evidence

Appraisal begins only after observable evidence has been stored.

### 4.1 Event representation

An event is:

$$
E_t=
\left(
c_t,
a_t,
x_t,
\widehat{\nu}_t,
g_t,
p_t
\right)
$$

where:

- $c_t$ is the semantic context.
- $a_t$ is the action attempted.
- $x_t$ is the event snapshot embedding.
- $\widehat{\nu}_t$ is the expected outcome before acting.
- $g_t$ is the task or goal.
- $p_t$ is event provenance.

### 4.2 Evidence item

Each outcome evidence item is represented as:

$$
o_{tj}
=
\left(
d_{tj},
m_{tj},
q_{tj},
s_{tj},
p_{tj}
\right)
$$

where:

- $d_{tj}\in\{-1,+1\}$ is the evidence direction.
- $m_{tj}\in[0,1]$ is the observed magnitude.
- $q_{tj}\in[0,1]$ is source reliability.
- $s_{tj}$ identifies the evaluator or source.
- $p_{tj}$ contains provenance.

A positive direction supports success. A negative direction supports failure.

Evidence sharing the same source and provenance must be deduplicated before aggregation.

### 4.3 Success and failure confidence

Let $S_t$ contain deduplicated success evidence and $F_t$ contain deduplicated failure evidence.

Define:

$$
c_t^{\text{success}}
=
1-
\prod_{j\in S_t}
\left(
1-q_{tj}m_{tj}
\right)
$$

$$
c_t^{\text{failure}}
=
1-
\prod_{j\in F_t}
\left(
1-q_{tj}m_{tj}
\right)
$$

Both values lie in $[0,1]$.

### 4.4 Evidence quality

Overall evidence quality is:

$$
q_t^{\text{evidence}}
=
\operatorname{clip}
\left(
q_t^{\text{source}}
\cdot
q_t^{\text{provenance}}
\cdot
q_t^{\text{corroboration}}
\cdot
q_t^{\text{integrity}},
0,
1
\right)
$$

Evidence quality must decrease when:

- provenance is missing;
- evaluators disagree;
- evidence is duplicated;
- an evaluator is unreliable;
- the observation is later corrected;
- the outcome cannot be reproduced.

### 4.5 Outcome valence

Outcome valence is:

$$
\nu_t
=
\operatorname{clip}
\left(
c_t^{\text{success}}
-
c_t^{\text{failure}},
-1,
1
\right)
$$

Positive values indicate better outcomes. Negative values indicate worse outcomes.

Valence is based on environmental outcome evidence, not on whether the event agrees with identity.

### 4.6 Disputed outcomes

Let $\theta_c$ be the confirmation threshold and $m_c$ the required confidence margin.

A success direction is confirmed when:

$$
c_t^{\text{success}}\geq\theta_c
\quad\land\quad
c_t^{\text{success}}-c_t^{\text{failure}}\geq m_c
$$

A failure direction is confirmed when:

$$
c_t^{\text{failure}}\geq\theta_c
\quad\land\quad
c_t^{\text{failure}}-c_t^{\text{success}}\geq m_c
$$

If neither condition holds, the outcome is unconfirmed or disputed.

Disputed evidence must not create wound, trauma, or identity-ledger updates until it is resolved.

---

## 5. Identity State

### 5.1 Identity ledgers

Identity contains two distinct ledgers:

$$
I_t=(P_t,N_t,s_t)
$$

where:

- $P_t$ contains positive self-facts.
- $N_t$ contains negative contrastive self-facts.
- $s_t$ is the current identity vector.

A positive self-fact represents behavior the agent identifies with and should reinforce.

A negative contrastive self-fact represents behavior, failure patterns, or external histories that define a boundary the agent should not reproduce.

Each ledger entry contains:

```text
IdentityFact
├── fact identifier
├── text
├── embedding
├── weight
├── source event
├── outcome evidence
├── provenance
├── created time
├── configuration version
├── correction state
└── deletion state
```

### 5.2 Contrastive identity vector

Let $f(\cdot)$ be the registered embedding function.

Define weighted ledger means:

$$
\overline{P}_t
=
\frac{
\sum_{p\in P_t}w_p f(p)
}{
\sum_{p\in P_t}w_p+\epsilon
}
$$

$$
\overline{N}_t
=
\frac{
\sum_{n\in N_t}w_n f(n)
}{
\sum_{n\in N_t}w_n+\epsilon
}
$$

The identity vector is:

$$
s_t
=
\operatorname{normalize}
\left(
\overline{P}_t
-
\mu\overline{N}_t
\right)
$$

where $\mu\geq0$ controls contrastive boundary strength.

Corrected, expired, or deleted facts must be excluded before recomputing $s_t$.

### 5.3 Identity alignment

For event embedding $x_t$:

$$
\alpha_t
=
\cos
\left(
x_t,s_t
\right)
$$

where:

$$
\alpha_t\in[-1,1]
$$

Interpretation:

- $\alpha_t>0$ means the event agrees with current identity.
- $\alpha_t<0$ means the event conflicts with current identity.
- $\alpha_t\approx0$ means the event is identity-neutral.

Identity alignment must be computed from the identity that existed before the outcome is incorporated.

---

## 6. Appraisal Function

The appraisal vector is:

$$
A_t
=
\left[
\alpha_t,
\nu_t,
\delta_t,
\sigma_t,
\rho_t
\right]
$$

where:

| Dimension | Range | Meaning |
|---|---:|---|
| $\alpha_t$ | $[-1,1]$ | Identity alignment |
| $\nu_t$ | $[-1,1]$ | Outcome valence |
| $\delta_t$ | $[-1,1]$ | Expectation violation |
| $\sigma_t$ | $[0,1]$ | Salience |
| $\rho_t$ | $[0,1]$ | Recurrence |

### 6.1 Expectation violation

Let $\widehat{\nu}_t$ be the predicted outcome before the action.

Then:

$$
\delta_t
=
\operatorname{clip}
\left(
\nu_t-\widehat{\nu}_t,
-1,
1
\right)
$$

Interpretation:

- $\delta_t>0$ means the result was better than expected.
- $\delta_t<0$ means the result was worse than expected.
- $\delta_t\approx0$ means the outcome matched expectation.

### 6.2 Recurrence

Recurrence is calculated only from semantically similar, eligible historical events.

Let:

$$
n_t^{-}
=
\sum_{j<t}
\mathbf{1}
\left[
\operatorname{sim}
\left(
(c_j,a_j),(c_t,a_t)
\right)
\geq\theta_{\text{rec}}
\right]
q_j^{\text{evidence}}
c_j^{\text{failure}}
\exp
\left(
-\kappa_{\text{rec}}(t-j)
\right)
$$

The normalized recurrence value is:

$$
\rho_t
=
1-
\exp
\left(
-\frac{n_t^{-}}{\tau_\rho}
\right)
$$

The raw effective recurrence count $n_t^{-}$ must also be retained because wound and trauma thresholds may depend on explicit recurrence counts.

Events from unrelated contexts must not contribute to recurrence merely because they share the same outcome label.

### 6.3 Salience

Salience is:

$$
\sigma_t
=
\operatorname{clip}
\left(
s_\nu|\nu_t|
+
s_\delta|\delta_t|
+
s_\alpha(-\alpha_t)^{+}
+
s_c
\max
\left(
c_t^{\text{success}},
c_t^{\text{failure}}
\right),
0,
1
\right)
$$

where:

$$
x^{+}=\max(0,x)
$$

Salience may depend on identity, but it must not replace semantic relevance during retrieval.

### 6.4 Determinism requirement

Given identical:

- identity ledgers;
- event;
- evidence;
- history;
- embeddings;
- configuration;
- model versions;

the appraisal function must reproduce the same $A_t$ within registered numerical tolerance.

---

## 7. Continuous Affective State

The architecture does not store emotion as a single categorical variable.

It computes a continuous functional state:

$$
Z_t
=
\left[
R_t^{\text{repeat}},
R_t^{\text{avoid}},
U_t^{\text{identity}},
F_t^{\text{retention}},
\Delta_t^{\text{expectation}}
\right]
$$

Operational labels are derived afterward.

### 7.1 Repeat evidence

Repeat evidence represents how strongly an action should be reinforced after confirmed success:

$$
R_t^{\text{repeat}}
=
c_t^{\text{success}}
\operatorname{sigmoid}
\left(
w_\nu^r\nu_t^{+}
+
w_\alpha^r\alpha_t^{+}
+
w_\delta^r\delta_t^{+}
+
w_\sigma^r\sigma_t
-
b_r
\right)
$$

Interpretation:

- positive valence increases reinforcement;
- identity-aligned success creates stronger reinforcement;
- unexpected success creates a larger update;
- salient success is consolidated more strongly;
- unconfirmed success produces little repeat evidence.

### 7.2 Avoidance evidence

Avoidance evidence represents how strongly an action should be avoided after confirmed failure:

$$
R_t^{\text{avoid}}
=
c_t^{\text{failure}}
\operatorname{sigmoid}
\left(
w_\nu^a(-\nu_t)^{+}
+
w_\alpha^a(-\alpha_t)^{+}
+
w_\delta^a(-\delta_t)^{+}
+
w_\sigma^a\sigma_t
+
w_\rho^a\rho_t
-
b_a
\right)
$$

Interpretation:

- negative valence increases avoidance;
- identity conflict increases corrective pressure;
- unexpectedly bad outcomes increase avoidance;
- salience increases severity;
- recurring similar failures accumulate avoidance evidence.

The same confirmed failure may produce different avoidance evidence for agents with different identities.

### 7.3 Tail gates

Define the positive-tail gate:

$$
G_t^{+}
=
\operatorname{sigmoid}
\left(
k_{+}
\left[
R_t^{\text{repeat}}-\theta_{+}
\right]
\right)
$$

Define the negative-tail gate:

$$
G_t^{-}
=
\operatorname{sigmoid}
\left(
k_{-}
\left[
R_t^{\text{avoid}}-\theta_{-}
\right]
\right)
$$

The gates are continuous. Ledger thresholds determine whether an actual identity update occurs.

### 7.4 Identity-update pressure

The signed identity-update pressure is:

$$
U_t^{\text{identity}}
=
\operatorname{clip}
\left(
G_t^{+}R_t^{\text{repeat}}
-
G_t^{-}R_t^{\text{avoid}},
-1,
1
\right)
$$

Interpretation:

- $U_t^{\text{identity}}>0$ creates positive self-fact pressure.
- $U_t^{\text{identity}}<0$ creates negative contrastive pressure.
- values near zero do not alter identity;
- isolated failures may alter policy without altering identity;
- wounds and trauma should create stronger negative pressure;
- only extreme or strongly confirmed successes should enter the positive identity ledger.

### 7.5 Retention pressure

The continuous retention floor is:

$$
F_t^{\text{retention}}
=
\operatorname{clip}
\left(
q_t^{\text{evidence}}
\left[
\phi_rR_t^{\text{repeat}}
+
\phi_aR_t^{\text{avoid}}
+
\phi_u
\left|
U_t^{\text{identity}}
\right|
+
\phi_\rho
c_t^{\text{failure}}
\rho_tG_t^{-}
\right],
0,
F_{\max}
\right)
$$

where $F_{\max}<1$ prevents memories from becoming permanently undeletable.

Interpretation:

- ordinary events receive little or no floor;
- success receives protection through repeat evidence;
- isolated failure receives limited corrective protection;
- wound receives additional protection through identity pressure;
- trauma receives strong protection through severity and recurrence;
- unreliable evidence reduces protection;
- deleted or invalidated evidence can set evidence quality to zero.

Retention controls survival and representation fidelity. It does not control semantic eligibility.

Therefore:

$$
F_t^{\text{retention}}
\not\Rightarrow
i\in C(q)
$$

### 7.6 Expectation transition

The adaptive expectation-learning rate is:

$$
\eta_t^{\text{expectation}}
=
\operatorname{clip}
\left(
\eta_0
+
\eta_\sigma\sigma_t
+
\eta_\rho\rho_t
+
\eta_u
\left|
U_t^{\text{identity}}
\right|,
0,
\eta_{\max}
\right)
$$

The expectation update is:

$$
\Delta_t^{\text{expectation}}
=
q_t^{\text{evidence}}
\eta_t^{\text{expectation}}
\delta_t
$$

For the relevant context-action key $k_t=(c_t,a_t)$:

$$
\widehat{\nu}_{t+1}(k_t)
=
\operatorname{clip}
\left(
\widehat{\nu}_t(k_t)
+
\Delta_t^{\text{expectation}},
-1,
1
\right)
$$

Expectation updates must remain local to semantically related context-action pairs.

A salient outcome from one task must not change expectations for unrelated tasks.

---

## 8. Operational Outcome Regions

Success, failure, wound, and trauma are derived regions of appraisal and affective state.

They are not independently assigned labels.

### 8.1 Success and injury intensity

Define success intensity:

$$
S_t
=
\operatorname{clip}
\left(
\zeta_rR_t^{\text{repeat}}
+
\zeta_u
\left(
U_t^{\text{identity}}
\right)^{+}
+
\zeta_\sigma\sigma_t,
0,
1
\right)
$$

Define injury intensity:

$$
J_t
=
\operatorname{clip}
\left(
\omega_aR_t^{\text{avoid}}
+
\omega_u
\left(
-U_t^{\text{identity}}
\right)^{+}
+
\omega_\sigma\sigma_t
+
\omega_\rho\rho_t,
0,
1
\right)
$$

The registered weights for each equation should sum to at most one.

### 8.2 Extreme consequence evidence

Let:

$$
\chi_t\in[0,1]
$$

represent directly observed extreme consequence evidence, such as:

- irreversible data loss;
- safety violation;
- unauthorized external action;
- severe financial impact;
- unrecoverable task corruption;
- explicitly registered catastrophic failure.

The value of $\chi_t$ must come from observable evidence and registered rules, not from emotional wording.

### 8.3 Region priority

Regions are evaluated in the following priority:

1. disputed or unconfirmed;
2. confirmed success;
3. confirmed failure;
4. wound;
5. trauma.

Trauma overrides wound. Wound overrides ordinary failure.

### 8.4 Ordinary

An event is ordinary when no outcome direction is sufficiently confirmed:

$$
\text{ordinary}
\iff
d_t^{\text{confirmed}}=\varnothing
$$

An ordinary event may still be stored and may still update a low-confidence expectation, but it must not update identity.

### 8.5 Success

An event is success when:

$$
\text{success}
\iff
d_t^{\text{confirmed}}=+1
$$

Success becomes identity-changing only when the positive-tail and ledger thresholds are crossed.

### 8.6 Failure

An event is an isolated failure when:

$$
\text{failure}
\iff
d_t^{\text{confirmed}}=-1
\land
\neg\text{wound}
\land
\neg\text{trauma}
$$

An isolated failure updates avoidance policy but normally does not rewrite identity.

### 8.7 Wound

A confirmed failure is a wound when:

$$
J_t\geq\theta_W
$$

and at least one of the following holds:

$$
(-\alpha_t)^{+}\geq\theta_{\alpha W}
$$

$$
n_t^{-}\geq n_W
$$

$$
\chi_t\geq\theta_{\chi W}
$$

A wound represents a severe corrective event that conflicts with identity, recurs, or produces severe observable consequences.

### 8.8 Trauma

A confirmed failure is trauma when either:

$$
\chi_t\geq\theta_{\chi T}
$$

or:

$$
J_t\geq\theta_T
\land
n_t^{-}\geq n_T
$$

with:

$$
\theta_T>\theta_W
\qquad\text{and}\qquad
n_T>n_W
$$

The initial controlled design uses:

$$
n_W=2
\qquad\text{and}\qquad
n_T=3
$$

This permits an extreme single event to enter trauma while requiring recurrence for ordinary severe failures.

### 8.9 Exclusivity

Every event receives exactly one operational label:

$$
L_t
\in
\{
\text{ordinary},
\text{success},
\text{failure},
\text{wound},
\text{trauma}
\}
$$

The continuous values $A_t$, $Z_t$, $S_t$, and $J_t$ must always be stored alongside the label.

---

## 9. Identity Transition

### 9.1 Positive ledger entry

Define positive snapshot weight:

$$
w_t^{+}
=
q_t^{\text{evidence}}
G_t^{+}
R_t^{\text{repeat}}
$$

A positive fact is added when:

$$
U_t^{\text{identity}}\geq\theta_P
$$

Then:

$$
P_{t+1}
=
P_t
\cup
\left\{
(x_t,w_t^{+},p_t)
\right\}
$$

### 9.2 Negative ledger entry

Define negative snapshot weight:

$$
w_t^{-}
=
\operatorname{clip}
\left(
q_t^{\text{evidence}}
G_t^{-}
R_t^{\text{avoid}}
\left(
1+\kappa_\rho\rho_t
\right),
0,
w_{\max}^{-}
\right)
$$

A negative contrastive fact is added when:

$$
U_t^{\text{identity}}\leq-\theta_N
$$

Then:

$$
N_{t+1}
=
N_t
\cup
\left\{
(x_t,w_t^{-},p_t)
\right\}
$$

### 9.3 No identity transition

If neither ledger threshold is crossed:

$$
P_{t+1}=P_t
\qquad
N_{t+1}=N_t
$$

The action policy may still change.

### 9.4 Recalculation

After a valid ledger transition:

$$
s_{t+1}
=
\operatorname{normalize}
\left(
\overline{P}_{t+1}
-
\mu\overline{N}_{t+1}
\right)
$$

Identity must be recomputed from active ledger entries instead of being mutated irreversibly.

This allows correction, deletion, expiry, and replay.

---

## 10. Policy Transition

Identity and action policy are separate.

For semantic context-action key $k_t=(c_t,a_t)$, maintain:

- repeat evidence ledger $L_t^{\text{repeat}}(k_t)$;
- avoidance evidence ledger $L_t^{\text{avoid}}(k_t)$.

### 10.1 Repeat transition

$$
L_{t+1}^{\text{repeat}}(k_t)
=
\xi_p
L_t^{\text{repeat}}(k_t)
+
q_t^{\text{evidence}}
R_t^{\text{repeat}}
$$

### 10.2 Avoidance transition

$$
L_{t+1}^{\text{avoid}}(k_t)
=
\xi_p
L_t^{\text{avoid}}(k_t)
+
q_t^{\text{evidence}}
R_t^{\text{avoid}}
$$

where $\xi_p\in[0,1]$ controls policy-evidence persistence.

### 10.3 Memory-derived action evidence

The signed memory contribution is:

$$
Q_t^{\text{memory}}(k)
=
L_t^{\text{repeat}}(k)
-
L_t^{\text{avoid}}(k)
$$

Positive values support repeating the action. Negative values support avoiding it.

This evidence may influence action selection, but it must remain traceable to its source episodes.

---

## 11. Retention and Forgetting

### 11.1 Identity-conditioned retention

For memory age $\tau$:

$$
r_t^{\text{identity}}(\tau)
=
\exp
\left[
-\lambda
\left(
1-\alpha_t
\right)
\tau
\right]
$$

### 11.2 Emotion-aware retention

$$
r_t^{\text{emotion}}(\tau)
=
F_t^{\text{retention}}
+
\left(
1-F_t^{\text{retention}}
\right)
r_t^{\text{identity}}(\tau)
$$

### 11.3 Meaning of retention

Retention may control:

- probability of survival;
- compression level;
- embedding fidelity;
- summary fidelity;
- archival priority;
- reconstruction quality.

It must not control:

- semantic eligibility;
- access permissions;
- deletion rights;
- provenance validity;
- direct reader inclusion.

### 11.4 Required retention modes

The implementation must expose at least five experimental modes:

| Mode | Description |
|---|---|
| `none` | No forgetting |
| `time` | Time-only decay |
| `identity` | Identity-conditioned decay |
| `reward` | Binary reward protection |
| `emotion` | Identity and continuous outcome-aware protection |

### 11.5 Reference reproduction target

Using the paper’s controlled configuration, the implementation should reproduce the following horizon-150 ordering:

| Mode | Ordinary | Success | Failure | Wound | Trauma |
|---|---:|---:|---:|---:|---:|
| None | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Time | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 |
| Identity | 0.186 | 0.978 | 0.148 | 0.024 | 0.014 |
| Reward | 0.011 | 0.258 | 0.258 | 0.258 | 0.345 |
| Emotion | 0.186 | 0.985 | 0.301 | 0.561 | 0.852 |

Exact numeric reproduction applies only to the registered reference fixture and configuration.

The general architectural invariant is:

$$
r^{\text{emotion}}_{\text{trauma}}
>
r^{\text{emotion}}_{\text{wound}}
>
r^{\text{emotion}}_{\text{failure}}
$$

at sufficiently long horizons.

---

## 12. Retrieval and Tree Attention

Retrieval follows:

$$
\text{query}
\rightarrow
\text{access filter}
\rightarrow
\text{semantic eligibility}
\rightarrow
\text{trajectory deduplication}
\rightarrow
\text{tree attention}
\rightarrow
\text{bounded tie-break}
\rightarrow
\text{reader}
$$

### 12.1 Access filter

Before semantic retrieval, exclude memories that are:

- deleted;
- expired;
- outside permissions;
- invalidated;
- missing required provenance;
- unavailable to the current agent.

### 12.2 Semantic eligibility

For query embedding $q$ and event-root embedding $h_i^{\text{root}}$:

$$
s_i^{\text{semantic}}
=
\cos
\left(
q,h_i^{\text{root}}
\right)
$$

The eligible set is:

$$
C(q)
=
\left\{
i:
s_i^{\text{semantic}}\geq\theta_{\text{relevance}}
\right\}
$$

A bounded Top-$K_{\text{candidate}}$ may be applied after the threshold.

Retention and salience must not add an event to $C(q)$.

### 12.3 Trajectory deduplication

Before final ranking, semantically equivalent states from the same trajectory must be grouped.

Only one representative should occupy a final retrieval slot unless the query explicitly requires a sequence.

The representative should preserve links to the other grouped nodes.

### 12.4 Tree attention

For each eligible event tree $i$, let $V_i$ be its nodes.

Each tree node has:

- content embedding $h_{iv}$;
- node type;
- depth;
- parent edge;
- provenance pointer.

The query-node attention logit is:

$$
\ell_{iv}
=
\frac{
(W_q q)^\top(W_k h_{iv})
}{
\sqrt{d}
}
+
b_{\text{type}(v)}
+
b_{\text{depth}(v)}
$$

Attention weights are:

$$
a_{iv}
=
\operatorname{softmax}_{v\in V_i}
\left(
\ell_{iv}
\right)
$$

The tree representation is:

$$
g_i
=
\sum_{v\in V_i}
a_{iv}W_vh_{iv}
$$

Typed-edge or ancestor masks may restrict attention so selected conclusions remain connected to their evidence and provenance.

Tree attention answers:

> Which parts of an already eligible event tree are useful for this query?

It does not answer:

> Which memories are globally important enough to bypass relevance?

### 12.5 Candidate score

After eligibility and deduplication:

$$
s_i^{\text{content}}
=
\lambda_r
\cos
\left(
q,h_i^{\text{root}}
\right)
+
\lambda_t
\cos
\left(
q,g_i
\right)
$$

Define a bounded retention and salience adjustment:

$$
b_i
=
\operatorname{clip}
\left(
\beta
\log
\left(
r_i^{\text{emotion}}+\epsilon
\right)
+
\gamma\sigma_i,
-B,
B
\right)
$$

The final score is:

$$
s_i^{\text{final}}
=
s_i^{\text{content}}
+
b_i
\qquad
i\in C(q)
$$

$B$ must remain small enough that the adjustment acts only as a tie-breaker.

### 12.6 Retrieval invariants

The retriever must guarantee:

1. ineligible memories never enter the reader;
2. salience cannot bypass the relevance threshold;
3. retention cannot bypass the relevance threshold;
4. deleted events remain unavailable;
5. trajectory duplicates do not fill Top-$K$;
6. selected tree nodes remain traceable to raw evidence;
7. every result exposes a selection trace.

### 12.7 Selection trace

Each retrieved memory should expose:

```text
RetrievalTrace
├── query identifier
├── semantic eligibility score
├── eligibility decision
├── trajectory group
├── selected tree nodes
├── tree-attention weights
├── retention value
├── bounded tie-break value
├── final score
├── provenance
└── exclusion reasons for rejected candidates
```

---

## 13. Reinforcement-Learning Integration

Reinforcement learning is downstream of evidence, appraisal, and memory.

RL must not assign emotional labels directly.

### 13.1 RL state

A policy may consume:

$$
\widetilde{s}_t
=
\left[
s_t^{\text{environment}},
s_t^{\text{identity}},
\widehat{\nu}_t,
g_t^{\text{memory}},
Q_t^{\text{memory}}
\right]
$$

where:

- $s_t^{\text{environment}}$ is the environment state.
- $s_t^{\text{identity}}$ is the identity vector.
- $\widehat{\nu}_t$ is the expected outcome.
- $g_t^{\text{memory}}$ is the tree-attended memory representation.
- $Q_t^{\text{memory}}$ is signed repeat/avoid evidence.

### 13.2 RL objective

For value-based learning:

$$
y_t
=
r_t^{\text{environment}}
+
\gamma_{\text{RL}}
\max_{a'}
Q_{\text{target}}
\left(
\widetilde{s}_{t+1},a'
\right)
$$

$$
\mathcal{L}_{\text{TD}}
=
\left[
Q_\theta
\left(
\widetilde{s}_t,a_t
\right)
-
y_t
\right]^2
$$

The environment reward remains the primary RL reward.

Affective-state values may be supplied as features or auxiliary targets, but they must not silently replace external outcome evidence.

### 13.3 Optional auxiliary losses

Identity sensitivity:

$$
\mathcal{L}_{\text{identity}}
=
\max
\left(
0,
\epsilon_Z
-
\left\|
Z(I^{(1)},E)
-
Z(I^{(2)},E)
\right\|_1
\right)
$$

Recurrence monotonicity:

$$
\mathcal{L}_{\text{recurrence}}
=
\sum_{k}
\max
\left(
0,
J_k-J_{k+1}+m_J
\right)
$$

Retrieval relevance:

$$
\mathcal{L}_{\text{relevance}}
=
\max
\left(
0,
s_{\text{irrelevant}}
-
s_{\text{relevant}}
+
m_R
\right)
$$

Auxiliary objectives must be reported separately from the RL return.

### 13.4 RL invariants

- Repeated confirmed success should increase support for the relevant action.
- Repeated confirmed failure should increase avoidance of the relevant action.
- Unrelated trauma must not alter an action solely through salience.
- Corrected evidence must alter subsequent policy evidence.
- Identity and policy updates must remain separately inspectable.

---

## 14. Correction, Expiry, and Deletion

Every derived state must be reversible through event replay.

### 14.1 Correction

When evidence is corrected:

1. mark the previous evidence version inactive;
2. append the corrected evidence;
3. recompute outcome confidence;
4. recompute appraisal;
5. recompute the affective state;
6. recompute the operational label;
7. recompute identity ledgers;
8. recompute policy ledgers;
9. recompute retention properties.

### 14.2 Expiry

Expiry may remove an event from retrieval and future recurrence calculations.

Expired identity facts must be excluded when identity is recomputed.

### 14.3 Deletion

Deletion must override retention:

$$
\text{deleted}(i)
\Rightarrow
i\notin C(q)
$$

No value of retention, wound, or trauma may make a deleted event retrievable.

### 14.4 Auditability

Every snapshot must record:

- raw evidence identifiers;
- identity version;
- embedding version;
- appraisal configuration;
- outcome-region configuration;
- retention configuration;
- retrieval configuration;
- policy version;
- creation time;
- correction history;
- deletion state.

---

## 15. Initial Reference Configuration

The following values are provisional engineering defaults, not psychological constants.

They must be stored in a versioned configuration and frozen before benchmark execution.

| Parameter | Initial value |
|---|---:|
| Confirmation threshold $\theta_c$ | 0.60 |
| Confirmation margin $m_c$ | 0.10 |
| Wound recurrence $n_W$ | 2 |
| Trauma recurrence $n_T$ | 3 |
| Wound injury threshold $\theta_W$ | 0.50 |
| Trauma injury threshold $\theta_T$ | 0.72 |
| Positive-tail threshold $\theta_+$ | 0.75 |
| Negative-tail threshold $\theta_-$ | 0.70 |
| Tail sharpness $k_+$ | 12 |
| Tail sharpness $k_-$ | 12 |
| Positive ledger threshold $\theta_P$ | 0.55 |
| Negative ledger threshold $\theta_N$ | 0.55 |
| Maximum retention floor $F_{\max}$ | 0.95 |
| Relevance threshold $\theta_{\text{relevance}}$ | 0.55 |
| Maximum ranking tie-break $B$ | 0.05 |

All weights and thresholds require:

- sensitivity analysis;
- ablation;
- multi-seed evaluation;
- natural-trajectory validation;
- reference-fixture reproduction.

Changing a parameter creates a new architecture configuration version.

---

## 16. Falsification Benchmarks

The architecture is rejected or revised when a required invariant fails.

### 16.1 Identity sensitivity

Construct two agents with deliberately contrastive identities and expose them to the same event and outcome evidence.

Required:

$$
A_t^{(1)}\neq A_t^{(2)}
$$

and:

$$
\left\|
Z_t^{(1)}-Z_t^{(2)}
\right\|_1
\geq\epsilon_Z
$$

The observable success and failure confidence must remain identical between agents.

Failure condition: identity changes the evidence itself, or the two contrastive identities produce indistinguishable appraisal and transition.

### 16.2 Determinism

Identical identity, event, evidence, history, models, and configuration must reproduce identical output within numerical tolerance.

Failure condition: repeated evaluation produces inconsistent labels, ledgers, retention, or retrieval traces.

### 16.3 Recurrence monotonicity

For three comparable failures:

$$
J_1<J_2<J_3
$$

Required progression:

```text
isolated failure → wound → trauma
```

Failure condition: recurrence decreases injury or causes trauma before its registered threshold without extreme-consequence evidence.

### 16.4 Isolated-failure separation

A confirmed isolated failure must satisfy:

$$
R_t^{\text{avoid}}>0
$$

while remaining below the negative identity-ledger threshold.

Failure condition: every ordinary failure rewrites identity.

### 16.5 Positive-tail correctness

An extreme confirmed success should:

- increase repeat evidence;
- enter the positive self-fact ledger;
- increase relevant action support;
- receive elevated retention.

Failure condition: extreme success does not affect any registered transition.

### 16.6 Negative-tail correctness

A wound or trauma should:

- increase avoidance evidence;
- enter the negative contrastive ledger when the threshold is crossed;
- receive stronger retention than isolated failure;
- remain correctable and deletable.

Failure condition: severe corrective memories are erased by identity conflict.

### 16.7 Retention reproduction

The reference fixture must reproduce the published horizon-150 values within a registered tolerance.

At minimum:

$$
r_{\text{trauma}}^{\text{emotion}}
>
r_{\text{wound}}^{\text{emotion}}
>
r_{\text{failure}}^{\text{emotion}}
$$

Failure condition: identity-only decay and emotion-aware retention remain indistinguishable for wound and trauma.

### 16.8 Relevance independence

Insert an irrelevant trauma with maximum retention and salience.

Required:

$$
\operatorname{sim}(q,i)<\theta_{\text{relevance}}
\Rightarrow
i\notin C(q)
$$

Failure condition: the irrelevant trauma enters retrieval because of its label, salience, or retention.

### 16.9 Trajectory deduplication

Create multiple highly similar snapshots from one trajectory.

Failure condition: duplicates occupy multiple final Top-$K$ slots when the query does not request a sequence.

### 16.10 Tree-attention traceability

For every retrieved event tree:

- attention weights must sum to one within tolerance;
- selected nodes must belong to the eligible tree;
- selected conclusions must link to evidence and provenance.

Failure condition: tree attention returns unsupported or cross-tree nodes.

### 16.11 Policy direction

After repeated confirmed success:

$$
Q_{t+1}^{\text{memory}}(c,a)
>
Q_t^{\text{memory}}(c,a)
$$

After repeated confirmed failure:

$$
Q_{t+1}^{\text{memory}}(c,a)
<
Q_t^{\text{memory}}(c,a)
$$

Failure condition: remembered negative outcomes increase preference for the failed action.

### 16.12 Correctability

Correcting or deleting evidence must produce the same state as replaying history without the invalid evidence.

Failure condition: identity, policy, or retention remains permanently changed after its only supporting evidence is removed.

### 16.13 Disputed-evidence safety

Conflicting high-confidence evidence must remain disputed until the confirmation-margin rule is satisfied.

Failure condition: disputed evidence creates wound, trauma, or irreversible identity changes.

### 16.14 Budget-matched evaluation

Compare retention modes under equal:

- storage budget;
- embedding fidelity;
- retrieval bandwidth;
- Top-$K$;
- reader context;
- latency budget.

Report:

- Recall@$K$;
- Precision@$K$;
- downstream task accuracy;
- action decision margin;
- memory usage;
- retrieval latency;
- correction accuracy.

No-forgetting must be identified as an oracle-like upper bound when budgets are not matched.

### 16.15 Multi-seed requirement

Synthetic and learned experiments must use multiple registered seeds.

Single-seed results may be used for debugging but not for general performance claims.

---

## 17. Component Boundaries

The implementation should maintain the following logical components:

| Component | Responsibility |
|---|---|
| Evidence store | Preserve observations and provenance |
| Identity ledger | Maintain positive and contrastive facts |
| Identity encoder | Produce the contrastive identity vector |
| Appraiser | Compute $A_t$ |
| State transition | Compute $Z_t$ |
| Outcome classifier | Derive operational regions |
| Policy ledger | Store repeat and avoidance evidence |
| Retention policy | Compute decay and retention floors |
| Memory topology | Maintain timestep sequence and event trees |
| Relevance gate | Create the semantic candidate set |
| Tree-attention retriever | Select evidence inside eligible trees |
| RL adapter | Expose memory and identity to a policy |
| Audit and replay engine | Apply correction, expiry, deletion, and recomputation |

A single component must not silently perform evidence collection, appraisal, retrieval, and policy updates together.

---

## 18. Implementation Order

Implementation begins only after this specification and its benchmark fixtures are frozen.

The implementation order is:

1. Define immutable event and evidence schemas.
2. Implement provenance and evidence deduplication.
3. Implement positive and negative identity ledgers.
4. Implement deterministic identity-vector construction.
5. Implement the appraisal vector.
6. Implement repeat and avoidance evidence.
7. Implement identity-update pressure.
8. Implement operational outcome regions.
9. Implement identity and policy transitions.
10. Implement retention modes.
11. Implement linked timestep memory and event trees.
12. Implement semantic eligibility and trajectory deduplication.
13. Implement tree attention.
14. Implement correction, deletion, replay, and audit traces.
15. Run falsification benchmarks.
16. Add reinforcement learning only after deterministic tests pass.
17. Run budget-matched and multi-seed evaluation.

---

## 19. Implementation Gate

The following has to be committed:

- this architecture document;
- versioned parameter configuration;
- deterministic synthetic fixtures;
- identity-counterfactual fixture;
- recurrence fixture;
- reference retention fixture;
- irrelevant-trauma retrieval fixture;
- correction and deletion fixture;
- expected outputs and tolerances.

The first implementation milestone is not an emotion-generating agent.

It is a deterministic mechanism that can prove:

1. identity changes appraisal;
2. outcome evidence remains independent of identity;
3. recurrence changes injury;
4. isolated failure updates policy without necessarily rewriting identity;
5. extreme outcomes update the appropriate identity ledger;
6. outcome salience changes retention;
7. semantic relevance remains the retrieval gate;
8. every state transition can be corrected, deleted, audited, and replayed.