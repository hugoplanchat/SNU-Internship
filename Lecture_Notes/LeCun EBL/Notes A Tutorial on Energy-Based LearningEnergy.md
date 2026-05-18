
**Core idea:** EBMs capture dependencies between variables by associating a scalar **energy** to each configuration of $(X, Y)$.

- **Inference** = find the $Y$ that minimizes $E(Y, X)$
- **Learning** = shape the energy function so that correct configurations get low energy, incorrect ones get high energy

**Why EBMs?** Probabilistic models require computing a normalization constant $Z$ (partition function) - often intractable. EBMs bypass this entirely since they make no probabilistic claim.

**Scope:** EBMs provide a unified framework covering discriminative models, generative models, conditional random fields, max-margin Markov networks, and manifold learning methods.

**Training** is guided by a **loss functional** (not to be confused with the energy function itself) which is minimized to produce the desired energy landscape.


>" Most probabilistic models can be viewed as special types of energy-based models in which the energy function satisfies certain normalizability conditions, and in which the loss function,  optimized by learning, has a particular form"

# SECTION 1



![[Pasted image 20260518100945.png]]

## 1.1 Energy-Based Inference

The model takes two sets of variables:
- **X** — observed variables (input, e.g. image pixels)
- **Y** — variables to predict (e.g. class label)

The model is an **energy function** $E(Y, X)$ that measures the *compatibility* between $X$ and $Y$.

**Convention:**
- Low energy → highly compatible configuration
- High energy → highly incompatible configuration

**Inference** = find the $Y$ that minimizes the energy given $X$:

$$Y^* = \arg\min_{Y \in \mathcal{Y}} E(Y, X)$$

**Important distinction:**
- The **energy function** $E(Y, X)$ is minimized by inference
- The **loss functional** (Section 2) is minimized by learning

When $\mathcal{Y}$ is small → enumerate all $Y$ and pick the smallest.
When $\mathcal{Y}$ is large → need a dedicated inference procedure (gradient descent, belief propagation, dynamic programming...).

## 1.2 What Questions Can a Model Answer?

Four types of tasks an EBM can handle:

1. **Prediction / classification / decision-making** : "Which $Y$ is most compatible with $X$?" → produce a single best answer
2. **Ranking**: "Is $Y_1$ or $Y_2$ more compatible with $X$?" → rank all possible answers, not just pick the best
3. **Detection**: "Is this $Y$ compatible with $X$?" → compare energy against a threshold (e.g. face detection)
4. **Conditional density estimation**: "What is $P(Y \mid X)$?" → output fed to another system or a human decision maker


## 1.3 Decision Making vs Probabilistic Modeling

**For pure decision-making**: you only need the correct $Y$ to have the lowest energy. The absolute values of energies don't matter.

**Problem when combining models**: energies are uncalibrated (arbitrary units) → two separately trained EBMs cannot be combined directly.

**Solution : Gibbs distribution**: convert energies to probabilities via:

$$P(Y \mid X) = \frac{e^{-\beta E(Y,X)}}{\int_{y \in \mathcal{Y}} e^{-\beta E(y,X)}}$$

$\beta$ = inverse temperature (positive constant). The denominator is the **partition function** $Z$, which normalizes to a valid distribution.

**The cost of normalization**: computing $Z$ requires integrating over all possible $Y$, which is intractable when $\mathcal{Y}$ is large or continuous. Use probabilistic modeling only when the application truly requires it, otherwise stick with plain EBM and avoid $Z$ entirely.

# SECTION 2 Energy-Based Training : Architecture and Loss Function 

## 2. Energy-Based Training: Architecture and Loss Function

Training an EBM = finding the energy function $E(W, Y, X)$ that produces the best $Y$ for any $X$, within a parameterized family:

$$\mathcal{E} = \{E(W, Y, X) : W \in \mathcal{W}\}$$

The **architecture** is the internal structure of $E(W, Y, X)$. No particular restriction on the nature of $X$, $Y$, $W$.

The quality of an energy function is measured by the **loss functional**,  $\mathcal{L}(W, \mathcal{S})$, minimized over the training set : 
$$\mathcal{S} = \{(X^i, Y^i) : i = 1 \dots P\}$$

$$W^* = \min_{W \in \mathcal{W}} \mathcal{L}(W, \mathcal{S})$$

For most cases, the loss functional is:

$$\mathcal{L}(E, \mathcal{S}) = \frac{1}{P} \sum_{i=1}^{P} L(Y^i, E(W, \mathcal{Y}, X^i)) + R(W)$$

$L$ is the **per-sample loss functional** and $R(W)$ is the **regularizer** encoding prior knowledge about which energy functions are preferable.

Building an EBM requires designing four components:

1. **The architecture**: internal structure of $E(W, Y, X)$
2. **The inference algorithm**: find $Y$ minimizing $E(W, Y, X)$ for a given $X$
3. **The loss function**: $\mathcal{L}(W, \mathcal{S})$ measures the quality of the energy function
4. **The learning algorithm**: find $W$ minimizing the loss over $\mathcal{E}$

The loss must assign low loss to energy functions that give the lowest energy to the correct answer and higher energy to all incorrect ones.

Three types of answers:

1. $Y^i$: the correct answer
2. $Y^{*i}$: the answer produced by the model (lowest energy)
3. $\bar{Y}^i$: the *most offending incorrect answer* — the incorrect answer with the lowest energy

The learning process must **push down** $E(W, Y^i, X^i)$ and **pull up** $E(W, \bar{Y}^i, X^i)$.



## 2.2.1 Energy Loss

$$L_\text{energy} = E(W, Y^i, X^i)$$

Minimizes directly the energy of the correct answer. Problem: nothing forces incorrect answers to have high energy. The network can set **all energies to zero** : loss is zero but the model predicts nothing useful. This is the **collapsed solution**.

Only safe with architectures where pushing $E(Y^i)$ down mechanically forces other energies up. Example: MSE regression $E = \|Y^i - G(W, X^i)\|^2$.

## 2.2.2 Generalized Perceptron Loss

$$L_\text{perceptron} = E(W, Y^i, X^i) - \min_{Y \in \mathcal{Y}} E(W, Y, X^i)$$

Always positive since the min is a lower bound on $E(Y^i)$.
Pushes $E(Y^i)$ down and pulls the model's answer up simultaneously.
Deficiency: no **energy gap** is enforced. The loss reaches zero as soon as $Y^i$ is the model's answer, even if all energies are nearly equal. The energy surface can still be flat.


## 2.2.3 Generalized Margin Losses

General form:

$$L_\text{margin} = Q_m(E(W, Y^i, X^i),\ E(W, \bar{Y}^i, X^i))$$

where $m > 0$ is the **margin** and $Q_m$ is a convex function whose gradient has a positive dot product with $[1, -1]$ in the region where $E(W, Y^i, X^i) + m > E(W, \bar{Y}^i, X^i)$.

The loss only activates when the correct answer is not lower by at least $m$:

$$E(W, Y^i, X^i) + m > E(W, \bar{Y}^i, X^i)$$

The gradient satisfies $\partial L / \partial E_C > 0$ and $\partial L / \partial E_I < 0$ simultaneously in the region where $E(W, Y^i, X^i) + m > E(W, \bar{Y}^i, X^i)$ : both the correct energy is pushed down and the incorrect energy is pushed up, with a guaranteed gap of $m$ between them once training converges.

## Hinge Loss and Log Loss

**Hinge Loss**: the most popular generalized margin loss:

$$L_\text{hinge}(W, Y^i, X^i) = \max\left(0,\ m + E(W, Y^i, X^i) - E(W, \bar{Y}^i, X^i)\right)$$

Penalizes the energy difference $E_C - E_I$ linearly when larger than $-m$.
Zero when the correct answer already wins by at least $m$. Only depends on energy **differences** : individual energy values are not constrained to take any particular value.

**Log Loss**: soft version of the hinge loss with infinite margin:

$$L_\text{log}(W, Y^i, X^i) = \log\left(1 + e^{E(W, Y^i, X^i) - E(W, \bar{Y}^i, X^i)}\right)$$
Equivalent to binary cross-entropy viewed from the EBM angle.

# LVQ2, MCE, Square-Square, Square-Exponential

**LVQ2 Loss**: zero-margin loss that saturates the ratio $E_C / E_I$ to $1 + \delta$:

$$L_\text{lvq2}(W, Y^i, X^i) = \min\left(1, \max\left(0, \frac{E(W, Y^i, X^i) - E(W, \bar{Y}^i, X^i)}{\delta\, E(W, Y^i, X^i)}\right)\right)$$

The saturation kicks in when $E_C / E_I \geq 1/(1-\delta) \approx 1 + \delta$, capping the contribution of outliers to a fixed cost of 1. Non-convex in $E_C$ and $E_I$.

**MCE Loss**: approximates the number of classification errors with a sigmoid instead of a step function (non-differentiable):

$$L_\text{mce}(W, Y^i, X^i) = \sigma\!\left(E(W, Y^i, X^i) - E(W, \bar{Y}^i, X^i)\right)$$

where $\sigma(x) = (1 + e^{-x})^{-1}$. No explicit margin, but still creates a gap between $E_C$ and $E_I$. Non-convex.

**Square-Square Loss**: treats $E_C$ and $E_I$ separately rather than through their difference:

$$L_\text{sq-sq}(W, Y^i, X^i) = E(W, Y^i, X^i)^2 + \left(\max(0,\, m - E(W, \bar{Y}^i, X^i))\right)^2$$

Penalizes large $E_C$ quadratically (push correct energy to zero) and small $E_I$ below $m$ quadratically (pull incorrect energy above $m$). Only suitable for energy functions bounded below by zero (e.g. architectures measuring a distance).

**Square-Exponential Loss**: same as square-square but replaces the quadratic contrastive term with an exponential:

$$L_\text{sq-exp}(W, Y^i, X^i) = E(W, Y^i, X^i)^2 + \gamma\, e^{-E(W, \bar{Y}^i, X^i)}$$

Infinite margin : pushes incorrect energies to infinity with exponentially decreasing force. $\gamma$ is a positive constant.

##  Negative Log-Likelihood Loss

Comes from the probabilistic formulation via the Gibbs distribution. Defined as:

$$L_\text{nll}(W, Y^i, X^i) = E(W, Y^i, X^i) + \mathcal{F}_\beta(W, \mathcal{Y}, X^i)$$

where $\mathcal{F}_\beta$ is the **free energy** of the ensemble:

$$\mathcal{F}_\beta(W, \mathcal{Y}, X^i) = \frac{1}{\beta} \log \int_{y \in \mathcal{Y}} \exp\!\left(-\beta E(W, y, X^i)\right) dy$$

The full loss functional averaged over the training set:

$$\mathcal{L}_\text{nll}(W, \mathcal{S}) = \frac{1}{P} \sum_{i=1}^{P} \left( E(W, Y^i, X^i) + \frac{1}{\beta} \log \int_{y \in \mathcal{Y}} e^{-\beta E(W, y, X^i)}\, dy \right)$$

The first term pushes $E_C$ down. The second term , the log partition function,  pulls all energies up, acting as the contrastive term. This is exactly the cross-entropy loss viewed through the EBM lens: the integral over $\mathcal{Y}$ is the partition function $Z$, which is intractable when $\mathcal{Y}$ is large or continuous.

## Section 3 : Simple Architectures

Three canonical architectures show how to build an EBM for standard tasks.

### 3.1 Regression

Energy function:

$$E(W, Y, X) = \frac{1}{2} \|G_W(X) - Y\|^2$$

Inference is trivial: $Y^* = G_W(X)$, the minimum energy is always zero.

With this architecture, energy loss, perceptron loss, and NLL loss are all equivalent. The perceptron contrastive term is zero (minimum is always zero). The NLL contrastive term is a Gaussian integral — constant with respect to $W$. So all three losses reduce to MSE:

$$\mathcal{L}_\text{energy}(W, \mathcal{S}) = \frac{1}{2P} \sum_{i=1}^P \|G_W(X^i) - Y^i\|^2$$

For linear $G_W$, one writes $G_W(X) = W^T \Phi(X)$ where $\Phi(X)$ is a vector of features $\phi_k(X)$. Training becomes convex least-squares. In deep models, $\phi$ is itself parameterized and learned — the loss is no longer convex in $W$.

### 3.2 Two-Class Classifier

$\mathcal{Y} = \{-1, +1\}$. Energy function:

$$E(W, Y, X) = -Y G_W(X)$$

$G_W(X)$ is a scalar discriminant. The convention $-YG_W(X)$ ensures low energy for the correct class: if $Y = +1$ we want $G_W(X) > 0$, if $Y = -1$ we want $G_W(X) < 0$.

Inference:

$$Y^* = \text{argmin}_{Y \in \{-1,+1\}} -Y G_W(X) = \text{sign}(G_W(X))$$

Applied to each loss with $E(W, Y, X) = -YG_W(X)$:

**Perceptron loss:**

$$\mathcal{L}_\text{perceptron}(W, \mathcal{S}) = \frac{1}{P} \sum_{i=1}^P \left(\text{sign}(G_W(X^i)) - Y^i\right) G_W(X^i)$$

SGD update rule: $W \leftarrow W + \eta \left(Y^i - \text{sign}(G_W(X^i))\right) \frac{\partial G_W(X^i)}{\partial W}$.

With linear $G_W(X) = W^T\Phi(X)$: classical perceptron learning rule.

**Hinge loss:**

$$\mathcal{L}_\text{hinge}(W, \mathcal{S}) = \frac{1}{P} \sum_{i=1}^P \max\left(0,\ m + 2Y^i G_W(X^i)\right)$$

With $G_W(X) = W^TX$ and regularizer $\|W\|^2$: linear SVM.

**NLL loss:**

$$\mathcal{L}_\text{nll}(W, \mathcal{S}) = \frac{1}{P} \sum_{i=1}^P \log\left(1 + e^{-2Y^i G_W(X^i)}\right)$$

Equivalent to the log loss (eq. 12). With linear $G_W(X) = W^T\Phi(X)$:

$$\mathcal{L}_\text{nll}(W, \mathcal{S}) = \frac{1}{P} \sum_{i=1}^P \log\left(1 + e^{-2Y^i W^T\Phi(X^i)}\right)$$

This is logistic regression.

### 3.3 Multiclass Classifier

$G_W(X)$ produces a vector $[g_1, \ldots, g_C]$, one scalar per class. The energy selects the component corresponding to $Y$ via a Kronecker delta:

$$E(W, Y, X) = \sum_{j=1}^C \delta(Y - j)\, g_j$$

In other words, $E_k = g_k$ — the energy of class $k$ is simply the $k$-th output of the network. Inference: take the index of the smallest component of $G_W(X)$, i.e. $\text{argmin}_k g_k$.

This is exactly the MNIST MLP with the convention $E_k = -z_k$: the perceptron, hinge, and NLL losses translate directly to the multiclass case.

### 3.4 Implicit Regression

In the previous architectures, the model maps $X \to Y$. But sometimes the relationship between $X$ and $Y$ cannot be expressed as a function — for example the constraint $X^2 + Y^2 = 1$, or tasks where multiple values of $Y$ are equally valid for a given $X$.

Solution: model the **constraint** that $X$ and $Y$ must satisfy. Both $X$ and $Y$ are passed through functions, and the energy measures the discrepancy between their outputs:

$$E(W, Y, X) = \frac{1}{2}\|G_X(W_X, X) - G_Y(W_Y, Y)\|^2$$

Canonical example: the **Siamese** architecture. Two instances of the same function $G_W$ receive $X_1$ and $X_2$. A binary label $Y$ encodes the constraint: if $Y = 0$, $G_W(X_1)$ and $G_W(X_2)$ should be equal (same person); if $Y = 1$, they should be different. The similarity metric is learned implicitly through $Y$ rather than through explicit supervision.

# 4. Latent Variable Architecture


![[Pasted image 20260518144457.png]]


In the standard EBM setup, the correct value of $Y$ is given for each training sample. But some tasks require energy functions that depend on hidden variables $Z$ whose correct value is never given during training. The inference process for a given $X$ and $Y$ then involves minimizing over these unseen variables:

$$E(Y, X) = \min_{Z \in \mathcal{Z}} E(Z, Y, X)$$

Such variables are called **latent variables**. They can be viewed as intermediate results on the way to finding the best output $Y$. Inference becomes a simultaneous minimization over both $Y$ and $Z$:

$$Y^* = \text{argmin}_{Y \in \mathcal{Y},\, Z \in \mathcal{Z}}\ E(Z, Y, X)$$

Latent variables are useful when a hidden characteristic of the process can be inferred from observations but cannot be predicted directly — for example the pose of an object in recognition, or the segmentation of words into phonemes in speech recognition.

### 4.1 An Example of Latent Variable Architecture — Face Detection

**Simple case:** a function $G_\text{face}(X)$ takes a small image window and outputs a small value if a face is present, large otherwise. The energy for the binary decision $Y \in \{0, 1\}$ is:

$$E(Y, X) = Y G_\text{face}(X) + (1 - Y)T$$

Inference: $Y = 1$ if $G_\text{face}(X) < T$, else $Y = 0$. No latent variable.

**Complex case — detection and localization:** apply $G_\text{face}$ to $K$ windows $X_1, \ldots, X_K$ of a large image. The position $Z \in \{1, \ldots, K\}$ of the face is unknown — it is the latent variable. The energy is:

$$E(Z, Y, X) = Y \left[\sum_{k=1}^K \delta(Z - k)\, G_\text{face}(X_k)\right] + (1 - Y)T$$

The Kronecker delta $\delta(Z - k)$ selects window $k$ when $Z = k$, so the sum reduces to $G_\text{face}(X_Z)$ — the energy of the selected window.

Inference minimizes jointly over $Y$ and $Z$: the model simultaneously finds the window with the lowest $G_\text{face}$ score (the location $Z^*$) and decides whether a face is present ($Y^*$). The position $Z$ is never supervised during training — it emerges from energy minimization.

### 4.2 Probabilistic Latent Variables

When the best value of $Z$ for a given $X$ and $Y$ is ambiguous, instead of minimizing over $Z$ one can marginalize — combining the contributions of all possible values of $Z$.

The joint conditional distribution over $Y$ and $Z$ is given by the Gibbs distribution:

$$P(Z, Y | X) = \frac{e^{-\beta E(Z, Y, X)}}{\int_{y \in \mathcal{Y},\, z \in \mathcal{Z}} e^{-\beta E(y, z, X)}}$$

Marginalizing over $Z$:

$$P(Y | X) = \frac{\int_{z \in \mathcal{Z}} e^{-\beta E(Z, Y, X)}}{\int_{y \in \mathcal{Y},\, z \in \mathcal{Z}} e^{-\beta E(y, z, X)}}$$

Finding the best $Y$ after marginalization:

$$Y^* = \text{argmin}_{Y \in \mathcal{Y}} -\frac{1}{\beta} \log \int_{z \in \mathcal{Z}} e^{-\beta E(z, Y, X)}$$

This is standard EBM inference with the energy redefined as the **free energy** of the ensemble $\{E(z, Y, X),\ z \in \mathcal{Z}\}$:

$$\mathcal{F}(Y) = -\frac{1}{\beta} \log \int_{z \in \mathcal{Z}} e^{-\beta E(z, Y, X)}$$

When $\beta \to \infty$ (zero temperature), the integral is dominated by the minimum of $E$ over $Z$, and marginalization reduces to the minimization inference of section 4.1.