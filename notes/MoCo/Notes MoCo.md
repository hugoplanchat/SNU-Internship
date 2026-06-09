
SSL Contrastive Learning  in Latent Space 

 # MoCo — Sections 1 & 2 (short summary)

## 1. Introduction
- In vision, supervised pre-training still dominates (unlike NLP), because the visual signal is**continuous and high-dimensional**, not discrete like words.
- Contrastive learning = **dictionary look-up**: a *query* (encoded image) must match its positive *key* and be dissimilar to negative keys.
- A good dictionary should be **(1) large** and **(2) consistent** (keys encoded in a comparable way).
- **MoCo**: dictionary = **queue** (decouples its size from the batch → large) + key encoder as a **momentum moving average** (→ consistent).
- Pretext task: *instance discrimination*. Result: MoCo **surpasses supervised pre-training** on 7 detection/segmentation tasks, and scales up (IG-1B).

## 2. Related Work
- Two independent axes: **pretext tasks** and **loss functions**; MoCo acts on the **loss-function** side.
- **Reconstruction** losses = fixed target; **contrastive** losses = target varies, based on similarities in the representation space (what MoCo uses).
- Pretext tasks: auto-encoders, colorization, patch ordering, clustering…
- Contrastive and pretext combine: instance discrimination ↔ NCE, CPC ↔ context auto-encoding, CMC ↔ colorization.


![[Pasted image 20260609192239.png]]

# 3. Method 

![[Pasted image 20260609192350.png]]![[Pasted image 20260609211025.png]]


## 3.1 Contrastive learning as dictionary look-up

MoCo frames contrastive learning as a **dictionary look-up**. An encoded **query** $q$ must match its single positive **key** $k_+$ among a set of keys $\{k_0, k_1, k_2, \dots\}$, and be dissimilar to all the others (the negatives). The loss is **InfoNCE**:

$$
\mathcal{L}_q = - \log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}
$$

a $(K{+}1)$-way softmax cross-entropy that tries to classify $q$ as $k_+$. The query is $q = f_q(x_q)$ and the keys are $k = f_k(x_k)$, produced by a **query encoder** $f_q$ and a **key encoder** $f_k$.

MoCo hypothesizes that good features come from a dictionary that is:
1. **large** — it samples the high-dimensional continuous visual space better (more negatives);
2. **consistent** — the keys should be encoded by the same or a very similar encoder, so the
   query-key comparisons are meaningful.

## 3.2 Momentum Contrast

**Dictionary as a queue.** The dictionary is a **FIFO queue** of encoded keys: the current mini-batch is enqueued, the oldest mini-batch is dequeued. This **decouples the dictionary size from the mini-batch size**, so it can be far larger than a batch (e.g. $K = 65536$) and is set independently. Removing the oldest keys is beneficial: they are the least consistent with the current encoder.

**How to update the key encoder $f_k$?** Two naive options fail, which motivates the momentum update.

### Why NOT option A: train $f_k$ by back-propagation
The loss depends on **all the keys in the queue**. To train $f_k$ by gradient descent, the gradient would have to **propagate back to every key in the queue**. But those keys were encoded at past steps by past versions of $f_k$; they are stored vectors with no computation graph. To do it properly you would have to **re-encode the entire queue at every step** and keep all their graphs, which is **intractable** and would destroy the whole point of the queue (reusing keys without recomputing them). So $f_k$ cannot be trained by back-propagation.

### Why NOT option B: copy $f_k = f_q$ at every step
Since $f_k$ can't be back-propagated, the naive fix is to **copy** the query weights into the key
encoder at each step ($f_k \leftarrow f_q$). This **fails** (it even diverges in experiments).
The reason is **consistency**: $f_q$ changes a lot at every gradient step, so a copied $f_k$ also
jumps rapidly. The keys in the queue were encoded over many past steps by these rapidly-changing encoders, so they become **mutually inconsistent** (each key measured with a different "ruler"). Comparing the query against an inconsistent set of keys gives a noisy, ill-defined signal, and training collapses.

### The solution: momentum update
$f_k$ is **not** trained by gradient at all. Its weights are an **exponential moving average** of the query encoder:

$$
\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q, \qquad m = 0.999
$$

Only $\theta_q$ is updated by back-propagation. With a large momentum ($m = 0.999$ works much better than $m = 0.9$), $f_k$ evolves **very slowly**, so all keys in the queue were encoded by **nearly the same encoder** → the dictionary stays **consistent**. The momentum encoder is a separate network (not just a copy), which is exactly what makes the queue usable.

**Summary of the three options:**

| Option | Update of $f_k$ | Result |
|---|---|---|
| A | back-propagation | intractable (gradient must reach the whole queue) |
| B | copy $f_k = f_q$ each step | $f_k$ jumps → inconsistent keys → diverges |
| C (MoCo) | momentum $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ | large **and** consistent → works |

## 3.3 Pretext task & technical details

MoCo uses **instance discrimination**: a query and a key form a positive pair if they are two augmented views of the **same** image, negatives otherwise. The encoder is a **ResNet** whose
output (after global average pooling, a 128-D FC) is **L2-normalized**, so dot products are
cosine similarities; the temperature is **$\tau = 0.07$**.

**Shuffling BN.** With BatchNorm, the model can "cheat" by exploiting intra-batch statistics leaked through BN (the batch stats act as a signature revealing which sample is the positive). MoCo fixes this with **shuffled BN**: the key mini-batch order is shuffled across GPUs before encoding (and unshuffled after), so the BN statistics for a query and its positive key come from different subsets, removing the leak.