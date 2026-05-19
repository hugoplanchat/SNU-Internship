

An **autoregressive** model generates a sequence token by token, using the already-generated tokens as input to predict the next one.

Formally, it factorizes the joint probability of a sequence as a product of conditional probabilities:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \ldots, x_{t-1})$$

At each step $t$, the model sees $x_1, \ldots, x_{t-1}$ and predicts $x_t$.


![[Pasted image 20260519111535.png]]

## Transformer Architecture

### Encoder / Decoder 

The Transformer has two sides:

**Encoder** : takes the input sequence and produces a rich contextual representation of each token. Each token's vector encodes its meaning in context : "cat" in "the cat sat" has a different representation than "cat" in "cat food" because self-attention lets each token aggregate information from its neighbors.

**Decoder** : generates the output sequence token by token (autoregressive). At each step $t$ it receives all previously generated tokens and predicts the next one. Cross-attention lets the decoder query the encoder's representations to decide which input tokens are relevant for generating the next output token.

**In the general case:**

Inputs = the sequence you want to understand (source sentence, image patches, past tokens)

Outputs = the sequence you are generating, shifted right by one position (teacher forcing during training)

Output probabilities = at each position $t$, a probability distribution over all possible tokens — take argmax for prediction, sample for generation
	
### Residual Connections

A residual connection adds the input of a sub-layer directly to its output:

$$\text{output} = x + \text{Sublayer}(x)$$

Instead of learning a full transformation $f(x)$, the network learns a correction $f(x) - x$. If the sub-layer learns nothing useful, it tends toward zero and lets $x$ pass through unchanged.

**Why it is necessary:** in backpropagation, the gradient flows through layers by chain rule multiplication. With $N$ layers each with derivative $\lambda < 1$:

$$\frac{\partial L}{\partial x_0} = \lambda^N \cdot \frac{\partial L}{\partial x_N}$$

This converges to zero exponentially fast with depth : the first layers stop learning. The residual connection adds $+1$ to the derivative:

$$\frac{\partial \text{output}}{\partial x} = 1 + \frac{\partial \text{Sublayer}(x)}{\partial x}$$

The $+1$ guarantees the gradient never vanishes regardless of what the sub-layer does. 


### Layer Normalization

After the residual addition, activations can have very different magnitudes across layers. LayerNorm re-centers and re-scales each vector to keep activations in a reasonable range throughout training.

For a vector $x \in \mathbb{R}^{d_\text{model}}$:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where:

$\mu = \frac{1}{d_\text{model}} \sum_i x_i$ : mean over the $d_\text{model}$ components of $x$

$\sigma = \sqrt{\frac{1}{d_\text{model}} \sum_i (x_i - \mu)^2}$ : standard deviation over the components

$\gamma, \beta$ : learned parameters (scale and shift), initialized to 1 and 0

**In PyTorch:** `nn.LayerNorm(d_model)`

**Full Add & Norm sub-layer:**

$$x \leftarrow \text{LayerNorm}(x + \text{Sublayer}(x))$$

# Attention

![[Pasted image 20260519114219.png]]


## 3.2.1 Scaled Dot-Product Attention

The attention function maps a query and a set of key-value pairs to an output. Queries, keys, values, and the output are all vectors. The output is a weighted sum of the values, where the weight of each value is determined by a compatibility function between the query and the corresponding key.

$$\boxed{\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}$$

### Dimensional Analysis

$$Q \in \mathbb{R}^{T \times d_k} \quad K \in \mathbb{R}^{T \times d_k} \quad V \in \mathbb{R}^{T \times d_v}$$

**Softmax** : Applied along the last dimension (over the $T$ keys for each query) 


**Remark : in practice with batching**

In practice, inputs have an extra batch dimension $B$ (number of examples processed in parallel):

$$Q \in \mathbb{R}^{B \times T \times d_k} \quad K \in \mathbb{R}^{B \times T \times d_k} \quad V \in \mathbb{R}^{B \times T \times d_v}$$

The computation is identical : PyTorch applies the matrix multiplications independently for each of the $B$ examples. The final output is $\mathbb{R}^{B \times T \times d_v}$.

### Additive vs Dot-Product Attention

Two standard attention functions exist:

**Dot-product attention** (this paper): $\text{score}(q, k) = q \cdot k$ — a simple dot product. Fast because it reduces to a single matrix multiplication $QK^\top$, which is highly optimized on GPU.

**Additive attention**: $\text{score}(q, k) = W_v^\top \tanh(W_q q + W_k k)$ — passes the concatenation of $q$ and $k$ through a single hidden layer feed-forward network. More flexible but much slower.

For small $d_k$ both perform similarly. For large $d_k$, dot-product without scaling is outperformed by additive attention because the dot products grow large and push the softmax into saturation. The scaling by $\frac{1}{\sqrt{d_k}}$ fixes this.

### Why divide by $\sqrt{d_k}$?

**Assumptions** : The components of $Q$ and $K$ are i.i.d. $\sim \mathcal{N}(0, 1)$.

**Objective**

We want $\forall i, j$:

$$\frac{(QK^\top)_{ij}}{\alpha} \sim \mathcal{N}(0, 1)$$

which is equivalent to finding $\alpha$ such that:

$$\text{Var}\left(\frac{(QK^\top)_{ij}}{\alpha}\right) = 1$$

**Derivation**

By the variance scaling property:

$$\frac{\text{Var}\left((QK^\top)_{ij}\right)}{\alpha^2} = 1 \implies \alpha^2 = \text{Var}\left((QK^\top)_{ij}\right)$$

By the Cauchy matrix multiplication formula:

$$(QK^\top)_{ij} = \sum_{l=1}^{d_k} Q_{il} K_{jl}$$

The terms $Q_{il} K_{jl}$ are independent. For each one:

$$\mathbb{E}[Q_{il} K_{jl}] = \mathbb{E}[Q_{il}]\mathbb{E}[K_{jl}] = 0$$

$$\text{Var}(Q_{il} K_{jl}) = \mathbb{E}[Q_{il}^2]\mathbb{E}[K_{jl}^2] = 1 \times 1 = 1$$

By additivity of variances of independent variables:

$$\text{Var}\left((QK^\top)_{ij}\right) = \sum_{l=1}^{d_k} \text{Var}(Q_{il} K_{jl}) = d_k$$

**Conclusion**

$$\alpha^2 = d_k \implies \alpha = \sqrt{d_k}$$

Without scaling, scores have std $\sqrt{d_k}$ — for large $d_k$ the softmax becomes a near one-hot vector with near-zero gradients. Dividing by $\sqrt{d_k}$ restores unit variance and keeps the softmax in a region with non-zero gradients.


## 3.2.2 Multi-Head Attention

Instead of applying a single attention with $d_\text{model}$ (dimensional queries), keys and values, project $h$ times into subspaces of dimension $d_k = d_\text{model}/h$, run attention in parallel on each projection, then concatenate and project back.

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

$$\text{head}_i = \text{Attention}(xW^Q_i,\ xW^K_i,\ xW^V_i)$$

Projection matrices:

$$W^Q_i \in \mathbb{R}^{d_\text{model} \times d_k} \quad W^K_i \in \mathbb{R}^{d_\text{model} \times d_k} \quad W^V_i \in \mathbb{R}^{d_\text{model} \times d_v} \quad W^O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$$

In the paper: $h = 8$, $d_\text{model} = 512$, $d_k = d_v = 64$.

### Full Chain from Input to Output

Starting from $x \in \mathbb{R}^{T \times d_\text{model}}$:

**Projections** (for each head $i$):

$$Q_i = xW^Q_i \in \mathbb{R}^{T \times d_k} \quad K_i = xW^K_i \in \mathbb{R}^{T \times d_k} \quad V_i = xW^V_i \in \mathbb{R}^{T \times d_v}$$

**Scaled dot-product attention per head:**

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{T \times d_v}$$

**Concatenation:**

$$\text{Concat}(\text{head}_1, \ldots, \text{head}_h) \in \mathbb{R}^{T \times h d_v} = \mathbb{R}^{T \times d_\text{model}}$$

since $h \times d_v = 8 \times 64 = 512 = d_\text{model}$.

**Output projection:**

$$\text{MultiHead}(x,x,x) = \text{Concat}(\ldots)\, W^O \in \mathbb{R}^{T \times d_\text{model}}$$

$W^O$ is square: $\mathbb{R}^{d_\text{model} \times d_\text{model}}$.

**Residual + LayerNorm:**

$$y = \text{LayerNorm}(x + \text{MultiHead}(x, x, x)) \in \mathbb{R}^{T \times d_\text{model}}$$

### Why Multiple Heads?

A single attention in $\mathbb{R}^{d_\text{model}}$ captures only one notion of similarity between tokens. With $h$ heads, each head projects into a different subspace and learns to capture a different type of relationship (syntactic, semantic, local, global). Nobody tells them what to learn, it emerges from random initialization and backpropagation.

## 3.2.3 Applications of Attention in the Transformer

Multi-head attention is used in three different ways in the Transformer.

**Encoder self-attention**

$Q$, $K$ and $V$ all come from the same source sequence $x_\text{encoder}$:

$$Q = x_\text{encoder} W^Q \quad K = x_\text{encoder} W^K \quad V = x_\text{encoder} W^V$$

Each token can attend to all other tokens in the source sequence. No mask — attention is bidirectional.

**Decoder masked self-attention**

$Q$, $K$ and $V$ all come from the same target sequence $x_\text{decoder}$:

$$Q = x_\text{decoder} W^Q \quad K = x_\text{decoder} W^K \quad V = x_\text{decoder} W^V$$

A causal mask is applied :  each token can only attend to positions $\leq t$. This enforces the autoregressive property: the model cannot look at future tokens during training.

**Decoder cross-attention**

Queries come from the decoder, keys and values come from the encoder output:

$$Q = x_\text{decoder} W^Q \quad K = x_\text{encoder} W^K \quad V = x_\text{encoder} W^V$$

Each decoder token can attend to all tokens of the source sequence. This is where the decoder "reads" the encoder representations to decide what to generate next.

## 3.3 Position-wise Feed-Forward Networks

In addition to the attention sub-layer, each block in the encoder and decoder contains a fully connected feed-forward network applied independently to each token position:

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

Two linear transformations with a ReLU in between. The same $W_1, W_2, b_1, b_2$ are applied to every position, but differ from layer to layer.

Dimensions in the paper: $d_\text{model} = 512$, inner dimension $d_\text{ff} = 2048 = 4 \times d_\text{model}$.

$$x \in \mathbb{R}^{T \times d_\text{model}} \xrightarrow{W_1 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}} \mathbb{R}^{T \times d_\text{ff}} \xrightarrow{\text{ReLU}} \mathbb{R}^{T \times d_\text{ff}} \xrightarrow{W_2 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}} \mathbb{R}^{T \times d_\text{model}}$$

The output dimension $\mathbb{R}^{T \times d_\text{model}}$ matches the input — the residual connection is valid.

**Full TransformerBlock:**

$$x' = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))$$

$$y = \text{LayerNorm}(x' + \text{FFN}(x'))$$

Input $x \in \mathbb{R}^{T \times d_\text{model}}$, output $y \in \mathbb{R}^{T \times d_\text{model}}$ 

**Remark — why $d_\text{ff} > d_\text{model}$?**

The first layer $W_1$ projects from $d_\text{model}$ to $d_\text{ff}$ — the vector is expanded into a larger space. The ReLU applies a non-linearity in this large space. The second layer $W_2$ projects back to $d_\text{model}$.

In the larger $d_\text{ff}$-dimensional space, the ReLU can selectively activate or deactivate many more directions than in $d_\text{model}$ dimensions. This is where the network stores and transforms information :  attention aggregates information across tokens, the FFN processes each token individually in this enriched space.

In modern LLMs, the FFN is interpreted as an **associative memory**: the weights $W_1$ and $W_2$ store key-value associations learned during training. The larger $d_\text{ff}$, the greater the memorization capacity.

The factor of 4 is empirical 

## 3.5 Positional Encoding

### The Problem

Attention is permutation-equivariant : if you permute the input tokens, the output is permuted the same way. The model has no notion of token order. "The cat sat" and "sat the cat" would produce the same representations, just in a different order.

### The Solution
	
Add a positional encoding vector to each token embedding before the attention layers:

$$x_\text{input} = \text{Embedding}(token) + PE_{pos}$$

### The Sinusoidal Formula

$$PE_{(pos,\, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

$$PE_{(pos,\, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

where $pos$ is the position in the sequence and $i$ is the dimension index. Each pair of dimensions $(2i, 2i+1)$ uses the same frequency $\omega_i = \frac{1}{10000^{2i/d_\text{model}}}$, one with sin and one with cos.

For example, with $d_\text{model} = 8$ and position $pos$:

$$PE_{pos} = \begin{bmatrix} \sin(\omega_0 \cdot pos) \\ \cos(\omega_0 \cdot pos) \\ \sin(\omega_1 \cdot pos) \\ \cos(\omega_1 \cdot pos) \\ \sin(\omega_2 \cdot pos) \\ \cos(\omega_2 \cdot pos) \\ \sin(\omega_3 \cdot pos) \\ \cos(\omega_3 \cdot pos) \end{bmatrix}$$

### Why Sin and Cos : Linear Representation of Relative Positions

The key property: $PE_{pos+k}$ is a linear function of $PE_{pos}$.

For a pair of dimensions $(2i, 2i+1)$ at frequency $\omega$, using the angle addition formulas:

$$\sin(\omega(pos+k)) = \sin(\omega \cdot pos)\cos(\omega k) + \cos(\omega \cdot pos)\sin(\omega k)$$


$$\cos(\omega(pos+k)) = \cos(\omega \cdot pos)\cos(\omega k) - \sin(\omega \cdot pos)\sin(\omega k)$$

In matrix form:

$$PE_{pos+k} = \underbrace{\begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix}}_{M_k} \cdot PE_{pos}$$

$M_k$ is a rotation matrix that depends only on the offset $k$, not on the absolute position $pos$. Moving from position $pos$ to position $pos+k$ is a rotation of angle $\omega k$ in the $(\sin, \cos)$ plane : the same rotation regardless of where you are in the sequence.

This is why the model can easily learn relative positions: "token at distance $k$" = apply $M_k$ to the positional encoding, which is a simple matrix multiplication.

