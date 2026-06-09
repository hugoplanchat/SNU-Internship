  

![[Pasted image 20260609150629.png]]
![[Pasted image 20260609150726.png]]
# NT-Xent loss (SimCLR)

For each image $k \in \{1, \dots, N\}$ in the batch, the two augmented views have indices $2k-1$ and $2k$. We use the loss on both directions of the positive pair and average over all $2N$ views:

  

$$

\mathcal{L}_{\text{NT-Xent}}

= \frac{1}{2N} \sum_{k=1}^{N} \Big[\, \ell(2k-1,\, 2k) + \ell(2k,\, 2k-1) \,\Big]

$$

  

Replacing $\ell(i,j)$ by its expression (softmax cross-entropy with temperature $\tau$):

  

$$

\mathcal{L}_{\text{NT-Xent}}

= \frac{1}{2N} \sum_{k=1}^{N} \left[

- \log \frac{\exp\!\big(\mathrm{sim}(z_{2k-1}, z_{2k})/\tau\big)}

{\displaystyle\sum_{m=1}^{2N} \mathbb{1}_{[m \neq 2k-1]}\, \exp\!\big(\mathrm{sim}(z_{2k-1}, z_{m})/\tau\big)}

\; - \;

\log \frac{\exp\!\big(\mathrm{sim}(z_{2k}, z_{2k-1})/\tau\big)}

{\displaystyle\sum_{m=1}^{2N} \mathbb{1}_{[m \neq 2k]}\, \exp\!\big(\mathrm{sim}(z_{2k}, z_{m})/\tau\big)}

\right]

$$

  

Replacing the cosine similarity by the $\ell_2$-normalized dot product $\mathrm{sim}(u, v) = \dfrac{u^\top v}{\lVert u \rVert \, \lVert v \rVert}$:

  

$$

\mathcal{L}_{\text{NT-Xent}}

= \frac{1}{2N} \sum_{k=1}^{N} \left[

- \log \frac{\exp\!\Big(\dfrac{z_{2k-1}^\top z_{2k}}{\tau\, \lVert z_{2k-1} \rVert\, \lVert z_{2k} \rVert}\Big)}

{\displaystyle\sum_{m=1}^{2N} \mathbb{1}_{[m \neq 2k-1]}\, \exp\!\Big(\dfrac{z_{2k-1}^\top z_{m}}{\tau\, \lVert z_{2k-1} \rVert\, \lVert z_{m} \rVert}\Big)}

\; - \;

\log \frac{\exp\!\Big(\dfrac{z_{2k}^\top z_{2k-1}}{\tau\, \lVert z_{2k} \rVert\, \lVert z_{2k-1} \rVert}\Big)}

{\displaystyle\sum_{m=1}^{2N} \mathbb{1}_{[m \neq 2k]}\, \exp\!\Big(\dfrac{z_{2k}^\top z_{m}}{\tau\, \lVert z_{2k} \rVert\, \lVert z_{m} \rVert}\Big)}

\right]

$$

  

where $z_i = g(f(\tilde{x}_i))$ is the projection of view $i$, $\tau$ is the temperature, and $\mathbb{1}_{[m \neq i]}$ excludes the anchor from its own denominator.

  

## Architecture of $f$ and $g$ in SimCLR

  

**Encoder $f$ : a ViT-Tiny backbone.** The input image is split into $4\times4$ patches (64 patches of dimension 48), linearly projected to $d_{\text{model}}=128$, prepended with a learnable `[CLS]` token and added to learnable positional embeddings. The sequence then goes through a pre-norm Transformer encoder of **6 layers** (4 heads, feed-forward dim 512, GELU), followed by a final LayerNorm. The representation is the `[CLS]` token:

  

$$h = f(\tilde{x}) \in \mathbb{R}^{128}$$

  

This is the **same ViT-Tiny backbone** shared across all methods (supervised ViT, MAE, CPC), which is what makes the benchmark a fair comparison.

  

**Projection head $g$ : a small MLP.** The representation $h$ is mapped to the space where the contrastive loss is applied by a two-layer MLP with a non-linearity:

  

$$z = g(h) = W^{(2)}\,\sigma\!\big(\text{BN}(W^{(1)} h)\big), \qquad z \leftarrow \frac{z}{\lVert z \rVert_2}$$

  

with $W^{(1)}: 128 \to 256$, ReLU activation $\sigma$, and $W^{(2)}: 256 \to 128$. The output $z$ is $\ell_2$-normalized so that the dot product in the NT-Xent loss is a cosine similarity.

  

**Important : $g$ is discarded after pre-training.** The NT-Xent loss is computed on $z$ (after $g$), but all downstream evaluation (linear probe and fine-tuning) uses $h$ (the encoder output, before $g$). The projection head is thrown away, because $g$ learns to be invariant to the augmentations and discards information (e.g. color, orientation) that is useful for downstream classification.



# The role of the dataset in SimCLR

Figure 4 :

![[Pasted image 20260609151228.png]]

Figure 5:

![[Pasted image 20260609151300.png]]


A defining insight of SimCLR is that **the contrastive prediction task is defined by data augmentation, not by the architecture**. Each image from the unlabeled dataset is transformed into two correlated views by a stochastic augmentation pipeline; these two views form the positive pair the encoder must pull together. Because the difficulty and the semantics of the pretext task are entirely controlled by *how the data is augmented*, the dataset augmentation policy plays the role that specialized architectures played in earlier methods. As the authors put it, prior work such as CPC (Oord et al., 2018) and Bachman et al. (2019) achieved global-to-local or adjacent-view prediction by *constraining the receptive field in the network architecture*, whereas SimCLR shows that "this complexity can be avoided by performing simple random cropping (with resizing)", which "conveniently decouples the predictive task from other components such as the neural network architecture" (Section 3). This is why the paper is titled *simple*: it "requires neither specialized architectures nor a memory bank" (Abstract).
).