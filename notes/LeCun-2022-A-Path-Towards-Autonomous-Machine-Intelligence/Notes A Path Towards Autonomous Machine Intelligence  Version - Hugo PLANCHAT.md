
## ARCHITECTURE



![[Pasted image 20260511035352.png]]

**Actor** : Orchestrator - configures all other modules depending on the task at hand

**Perception** : Estimates the current state of the world from sensor inputs

**World Model**  : Predicts possible future world states

**Cost module** : Measures the agent's "discomfort" as a scalar energy value
-Intrinsic Costis hard-wired and non-trainable - it captures the agent's basic drives (pain, pleasure, hunger) as an immediate energy value for the current state.
-Trainable Critic learns to predict future intrinsic costs from past experience, allowing the agent to anticipate discomfort before it happens rather than just reacting to it.

**Short-term memory** : Stores past, current and predicted future states along with their associated costs

**Actor** : Proposes action sequences and optimizes them to minimize the estimated cost

## MODE-1

![[Pasted image 20260511044542.png]]

*Notations :*

$x$ : *raw input (pixels, sensor signals...)* 
$s= \text{Enc}(x)$ : *abstract representation in a latent space*
$C$ : *Cost module*
$f = C(s)$ : Energy value 
$A$ : *Actor Module*

In **Mode-1**, the world model is not used to decide the action. The agent goes directly:

$$
x \rightarrow  \text{Enc}(x) \rightarrow s[0] \rightarrow A(s) \rightarrow a[0]
$$
The actor reacts instantly without imagining future states  - this is the reflex, System 1 behavior.

$\text{Pred}(s,a)$ is optional in Mode-1. LeCun mentions it can be used passively - after the action is taken, the world model predicts $s[1]$ and waits for the next real observation to adjust itself. But it plays no role in choosing the action.


## Mode-2 : Perception-Action Loop

![[Pasted image 20260511044635.png]]

*Notations :*

$T$ : Hyperparameter Planning horizon 



1. **Perception** : the perception module encodes the current observation into a world state $s[0] = P(x)$. The cost module computes and stores the immediate cost $C(s[0])$.

2. **Action proposal** : the actor proposes an initial sequence of actions $(a[0], \dots, a[t], \dots, a[T])$ to be evaluated by the world model.

3. **Simulation** : the world model recursively predicts the resulting world states:
   $$s[t+1] = \text{Pred}(s[t], a[t])$$

4. **Evaluation** : the cost module estimates the total cost over the predicted sequence:
   $$F(x) = \sum_{t=1}^{T} C(s[t])$$

5. **Planning** : the actor proposes a new action sequence with lower cost via gradient-based optimization. Steps 2-5 are iterated until convergence.
   The optimal sequence is denoted $(\check{a}[0], \dots, \check{a}[T])$.

6. **Acting** :  the actor sends the first action $\check{a}[0]$ (or first few actions ) to the effectors. The entire process repeats for the next episode.

7. **Memory** :  states and associated costs are stored in short-term memory to later train or adapt the critic.

This procedure is essentially what is known as Model-Predictive Control (MPC) with receding horizon in the optimal control literature. The difference with classical optimal control is that the world model and the cost function are learned. 

## FROM MODE-2 to MODE-1 : Learning New Skills 

![[Pasted image 20260511050953.png]]

*Notations:*

$D$ : divergence mesure (distance)

Mode-2 is powerful but expensive - it runs the world model repeatedly over a full horizon $T$ before every action. 
The idea here is to use Mode-2 as a teacher : run it once to produce optimal action sequences $(\check{a}[0], \dots, \check{a}[T])$, then train the policy module $A(s[t])$ to imitate those actions by minimizing
$D(\check{a}[t], A(s[t]))$. The result is a reactive policy that approximates Mode-2 quality at Mode-1 speed -this is **amortized inference**. The trained policy can then either act directly in Mode-1, or serve as a warm start to accelerate Mode-2 optimization.

## THE COST MODULE


![[Pasted image 20260511051620.png]]


The cost module outputs a single scalar called **energy** — it measures the level
of discomfort of the agent at a given world state $s[t]$.

$$f[t] = C(s[t])$$

It is composed of two sub-modules:

- **Intrinsic Cost** : hard-wired, non-trainable. Computes the immediate energy
  of the current state (pain, pleasure, hunger…). The ultimate goal of the agent
  is to minimize this over the long run.

- **Trainable Critic** : learns to predict future intrinsic costs from past experience,
  allowing the agent to anticipate discomfort before it happens.

Because both sub-modules are differentiable, the gradient of the energy can be
back-propagated through the world model and the actor, enabling planning and learning.

The total cost over a Mode-2 sequence is:

$$F(x) = \sum_{t=1}^{T} C(s[t])$$
## Training the Critic

![[Pasted image 20260511074505.png]]

During operation, the intrinsic cost module continuously writes triplets into short-term memory:
$(\tau, s_\tau, \text{IC}(s_\tau))$ — time, state, intrinsic energy.

The critic then reads two entries: a past state $s_\tau$ and the intrinsic cost at a later time $\text{IC}(s_{\tau+\delta})$, and trains to minimize:

$$||\text{IC}(s_{\tau+\delta}) - \text{TC}(s_\tau)||^2$$

**Goal:** given state $s_\tau$, predict future discomfort at $\tau + \delta$.
This allows the agent to anticipate bad outcomes during Mode-2 planning
without unrolling the world model all the way to the end.


## DESIGNING AND TRAINING WORLD MODEL


>"designing architectures and training paradigms for the world model constitute
the main obstacles towards real progress in AI over the next decades" 

![[Pasted image 20260511082528.png]]

Training the world model is a prototypical example of SSL 

Training World Model - predicting future representations of the state of the world

EBM : Energy-Based Models [[Notes A Tutorial on Energy-Based Learning]]

The system is a scalar-valued function F (x, y) that produces low energy values when x and
y are compatible and higher values when they are not.
The energy function produces low energy values around the data points, and higher energies away from the regions of high data density, as symbolized by the contour lines of the energy landscape.

Given two inputs:
- $x$ — the current observation (what the agent sees **now**)
- $y$ — a future observation (what the agent will see **later**)

Learn two encoders:
$$s_x = g_x(x) \qquad s_y = g_y(y)$$

These representations must satisfy two conditions simultaneously:

**Condition 1 — Informativeness**
$s_x$ and $s_y$ must capture as much information as possible about $x$ and $y$.
The encoder must not collapse everything into a constant vector just to make prediction easy.

**Condition 2 — Predictability**
$s_y$ must be easily predictable from $s_x$.
The world model must be able to go from the current representation to the future one.

**The tension between the two:**
- Optimize only condition 2 → encoder collapses ($s_x = s_y = 0$, trivially predictable but useless)
- Optimize only condition 1 → rich representations but impossible to predict

> JEPA is designed to find the right balance between informativeness and predictability.
> This is the **representation collapse problem** — the central challenge of all SSL methods.

>"What concepts could such an SSL system learn by being trained on video? Our hypoth-
esis is that a hierarchy of abstract concepts about how the world works could be acquired."

## HANDLING UNCERTAINTY WITH LATENT VARIABLES

**Latent Variable** : variable not directly observed - it exists in the model to represent hidden or unknown information

![[Pasted image 20260511084159.png]]

## Latent Variables in EBM

A **latent variable** $z$ is an input that is not observed but **inferred** by the model. It represents information about $y$ that cannot be extracted from $x$ alone.

In a temporal prediction scenario, $z$ captures **everything that cannot be predicted**
about the future from the past alone.

A **Latent-Variable EBM (LVEBM)** defines an energy over $(x, y, z)$:

$$E_w(x, y, z)$$

Given a pair $(x, y)$, the model infers the optimal latent variable:

$$\check{z} = \arg\min_{z \in \mathcal{Z}} E_w(x, y, z)$$

Which gives the **free energy** (energy with $z$ eliminated):

$$F_w(x, y) = \min_{z \in \mathcal{Z}} E_w(x, y, z) = E_w(x, y, \check{z})$$

> $z$ allows the model to represent **multiple plausible futures** from the same $x$ - without it, the world model can only output a single deterministic prediction.

$w$ : weights of the NN (trained)

## TRAINING EBM

## EBMs are not Probabilistic Models

Standard ML models $p(y|x)$ require computing a normalization constant:

$$p(y|x) = \frac{e^{-E(x,y)}}{Z(x)} \qquad Z(x) = \int e^{-E(x,y)} \, dy$$

Computing $Z(x)$ requires integrating over **all possible $y$** : intractable in high dimensions.

**EBMs sidestep this entirely.** The energy $E_w(x, y)$ is just a scalar measuring **compatibility** between $x$ and $y$. No probability, no normalization required.

You only need:
- $E_w(x, y)$ **low** → $x$ and $y$ are compatible
- $E_w(x, y)$ **high** → $x$ and $y$ are incompatible

> The energy is the fundamental object — not an implicit log-probability in disguise.

**Why this matters for JEPA:** predicting in latent space would require normalizing over all possible future embeddings if forced into a probabilistic framework — impossible.
EBMs avoid this problem completely.

training sample $(x,y)$

loss functional 
$$\mathcal{L}(x, y, F_w(x, y)) = \mathcal{L}(x, y, w)$$
## Preventing Collapse: $F_w(x, \hat{y}) > F_w(x, y)$

The core requirement for a well-trained EBM is:

$$F_w(x, y) < F_w(x, \hat{y}) \quad \forall \hat{y} \neq y$$

Pushing down $F_w(x, y)$ is easy - it suffices for the loss to be an increasing function of the energy with a lower bound.

The hard question is: **how do we ensure that $F_w(x, \hat{y})$ stays high?**
Without an explicit mechanism, the energy landscape may **collapse** - becoming flat and assigning the same low energy to all $y$, making the model useless.

Two approaches to prevent this:

- **Contrastive methods** — explicitly push up $F_w(x, \hat{y})$ using negative samples $\hat{y}$. Requires carefully chosen negatives, scales poorly with dimension.

- **Regularized methods** — add a term to the loss that minimizes the * volume* of the low-energy region, shrink-wrapping it around real data. No negative samples needed. JEPA uses this approach.

![[Pasted image 20260511092632.png]]![[Pasted image 20260511092643.png]]

## JEPA

>"JEPA is not generative in the sense that it cannot easily be used to predict y from x. It merely
capture the dependencies between x and y without explicitly generating predictions of y"

![[Pasted image 20260511093741.png]]

The two variables x and y are fed to two encoders producing two presentations sx and sy. These two encoders may be different. They are not required to possess the same architecture nor are they required to share their parameters. This allows x and y to be different in nature

JEPA have to : 
1. maximize the information content of sx about x
2. maximize the information content of sy about y
3. make sy easily predictable from sx
4. minimize the information content of the latent variable z used in the prediction.

**Criteria 1 & 2 — Informativeness of $s_x$ and $s_y$**
Prevent collapse by ensuring $s_x$ and $s_y$ carry as much information as possible about their inputs. Without these, the system could set $s_x = s_y = \text{const}$, making the energy flat over the entire input space.

**Criterion 3 — Predictability**
Enforced by the energy term $D(s_y, \tilde{s}_y)$. Ensures that $y$ is predictable from $x$ in representation space.

**Criterion 4 — Limiting the latent variable $z$**
Prevents a second type of collapse. Thought experiment:

> If $z$ has the same dimension as $s_y$, the predictor could learn to ignore $s_x$
> and simply copy $z$ onto its output: $\tilde{s}_y = z$.
> Then for any $s_y$, setting $\check{z} = s_y$ makes $D(s_y, \tilde{s}_y) = 0$ everywhere.
> → **Total collapse**: the energy surface is flat, the model learns nothing.

**Solution:** limit the information content of $z$ by making it:
- discrete
- low-dimensional
- sparse
- or noisy

This forces the predictor to rely on $s_x$ to predict $s_y$, and prevents $z$ from encoding all of $y$ by itself.

## VICREg

![[Pasted image 20260511100558.png]]

## H-JEPA

![[Pasted image 20260511101920.png]]

H-JEPA is an extension of JEPA that learns representations at **multiple levels** of abstraction simultaneously**, mimicking the hierarchical structure of human cognition.

Instead of a single encoder-predictor pair $(s_x, s_y)$, H-JEPA stacks multiple JEPA modules on top of each other:

- **Low-level JEPA** — predicts fine-grained representations (edges, textures, short-term motion)
- **Mid-level JEPA** — predicts more abstract representations (objects, trajectories)
- **High-level JEPA** — predicts high-level concepts (goals, intentions, long-term plans)

Each level predicts in its own representation space, at its own **time scale**. Higher levels operate over longer horizons and more abstract concepts.

This directly addresses one of the three core challenges LeCun identifies:
> *"How can machines learn to represent percepts and action plans in a hierarchical
> manner, at multiple levels of abstraction, and multiple time scales?"*


## HIERARCHICAL PLANNING 

![[Pasted image 20260511102410.png]]

Representations at multiple **levels of abstractions** by a cascade of encoders:
$$
s[0]=\text{Enc1}(x); \quad s2[0]=\text{Enc2}(s[0]);...
$$
**Level 2 (high)** — plans over long horizons, infers abstract "actions" $a2[2], a2[4]$ which are not real actions but **subgoals** (target states for level 1)

**Level 1 (low)** — takes those subgoals as cost targets $C(s[2]), C(s[4])$ and finds real low-level actions $a[0], a[1], \dots$ to achieve them

The process is sequential top-down, but ideally all levels are optimized **jointly**.

## HANDLING UNCERTAINTY 

![[Pasted image 20260512034339.png]]


Even with a perfect world model, predictions are uncertain for 5 reasons:

- **Aleatoric type 1** — the world is intrinsically stochastic (random events)
- **Aleatoric type 2** — the world is deterministic but chaotic (turbulence, weather)
- **Aleatoric type 3** — the world is partially observable
- **Epistemic type 1** — sensors give only partial information about the true state
- **Epistemic type 2/3/4** — the world model itself is imperfect (limited data, compute, capacity)

> **Aleatoric** = uncertainty in the world (irreducible)
> **Epistemic** = uncertainty in the model (reducible with more data/compute)

**Solution — latent variables + sampling** 

At each step, sample multiple values of $z$ to generate multiple plausible futures:
$$s[t+1] = \text{Pred}(s[t], a[t], z[t])$$

Problem: with $k$ discrete values of $z$ and $t$ steps → $k^t$ trajectories (exponential explosion).
Fix: directed search and pruning (e.g. Monte-Carlo Tree Search).

**Key insight:**
> Uncertainty is not in the model outputs — it lives entirely in $z$.
> The world model stays deterministic. All stochasticity is pushed into the latent variable.

##  Keeping Track of the State of the World

The world state should not be communicated as a full vector at each step,most of the world changes only slightly between two timesteps.
Instead, the world state is stored in a **writable associative memory** where only the affected parts are updated at each step.

**Key-Value Memory**

The memory stores pairs $(k_j, v_j)$ : key and value for each entry. Given a query $q$ (the current state), retrieval works in 3 steps:

$$\tilde{c}_j = \text{Match}(k_j, q) \tag{16}$$
$$c = \text{Normalize}(\tilde{c}) \tag{17}$$
$$\text{Mem}(q) = \sum_j c_j v_j \tag{15}$$

1. **Match** — compute similarity between query $q$ and each key $k_j$
2. **Normalize** — turn raw scores into weights that sum to 1
3. **Retrieve** — return a weighted sum of the stored values

**Writing** to memory updates existing entries:
$$v_j \leftarrow \text{Update}(r, v_j, c_j) = c_j r + (1 - c_j) v_j$$

where $r$ is the new value to write.

> This is the same mechanism as **attention** in Transformers (Q, K, V).
> It allows one-shot learning and smooth interpolation between stored states.

## Data Streams

How should an agent collect data to train its world model?

1. **Passive observation** — receives a sensor stream (video, audio…), no control over what it sees
2. **Active foveation** — can orient its sensors without affecting the environment (moving eyes/head)
3. **Passive agency** — watches another agent acting, learns causal effects without acting itself
4. **Active egomotion** — moves its sensors in the environment without significantly affecting the world
5. **Active agency** — acts on the environment and observes consequences → enables causal models,
   but introduces the **exploration-exploitation dilemma**

> **Open question:** how much can be learned from passive observation (modes 1, 2, 4),
> and how much requires active agency (mode 5)?

## DESINGING AND TRAINING THE ACTOR 


The actor has **3 roles**:
1. Infer optimal action sequences minimizing the cost (Mode-2)
2. Explore configurations of latent variables $z$ to plan under uncertainty
3. Train policy networks for Mode-1 reactive behavior

**Key insight:** there is no conceptual difference between an action $a$ and a latent
variable $z$ — both are variables the actor must explore to find good configurations.

**When the world model is well-behaved** → gradient-based optimization:
$$\min_{a[0],\dots,a[T]} \sum_{t=1}^{T} C(s[t])$$

**When not well-behaved** (discrete actions, discontinuities) → use:
- Dynamic programming
- Beam search
- Monte-Carlo Tree Search (MCTS)

**After optimization** → train a policy network $A(s[t])$ to imitate the optimal
sequence (Mode-2 → Mode-1 distillation, same as Section 3.1.3).
The policy can then act directly in Mode-1 or warm-start the next Mode-2 optimization.

## DESIGNING THE CONFIGURATOR 

The configurator is the main controller of the architecture : it takes input from all modules and modulates their parameters and connection graphs to configure the agent for the task at hand.

**2 reasons it exists:**

- **Hardware reuse** — one generic world model for all tasks, configured on the fly
- **Knowledge sharing** — knowledge from one task transfers to another with minor changes

Downside: the agent can only **focus on one task at a time**.

**How it configures each module:**

- **Perception** — primes it to extract task-relevant features
- **Predictor (low-level)** — configures local feature routing for short-term predictions
- **Predictor (high-level)** — adds tokens to Transformer blocks to modulate attention
- **Cost module** — sets subgoals by modulating weights $u_i, v_j$ of cost sub-modules

