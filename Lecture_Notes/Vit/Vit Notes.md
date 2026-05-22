
# Architecture :

![[Pasted image 20260521114543.png]]


# ViT — Vision Transformer (Dosovitskiy et al., 2021)

## Notation

| Symbol | Value (ViT-Tiny, CIFAR-10) | Description |
|--------|---------------------------|-------------|
| $H, W$ | 32, 32 | image height and width |
| $C$ | 3 | number of channels (RGB) |
| $P$ | 4 | patch size (P×P pixels) |
| $N$ | $HW/P^2 = 64$ | number of patches |
| $T$ | $N + 1 = 65$ | total number of tokens (patches + CLS) |
| $d_{\text{model}}$ | 128 | token embedding dimension |
| $L$ | 12 | number of Transformer blocks |
| $d_{\text{ff}}$ | 512 | MLP hidden dimension ($= 4 \times d_{\text{model}}$) |

---

## Architecture

### Equation 1 — Patch embedding

Each patch $x_p^i \in \mathbb{R}^{P^2 \cdot C}$ is a flattened patch of $P^2 \cdot C = 48$ raw pixel values.

The initial token sequence is built by concatenating the CLS token and the projected patches:

$$z_0 = \begin{bmatrix} x_{\text{class}} \\ x_p^1 \, E \\ x_p^2 \, E \\ \vdots \\ x_p^N \, E \end{bmatrix} + E_{\text{pos}} \quad \in \mathbb{R}^{T \times d_{\text{model}}}$$

where:
- $x_{\text{class}} \in \mathbb{R}^{d_{\text{model}}}$ : learnable CLS token (row 0)
- $E \in \mathbb{R}^{(P^2 \cdot C) \times d_{\text{model}}}$ : patch projection matrix (`nn.Linear(48, 128, bias=False)`)
- $E_{\text{pos}} \in \mathbb{R}^{T \times d_{\text{model}}}$ : learnable positional embeddings

---

### Equation 2 — MSA sub-layer (for each block $\ell = 1, \dots, L$)

$$z'_\ell = \text{MSA}\!\left(\text{LN}(z_{\ell-1})\right) + z_{\ell-1} \quad \in \mathbb{R}^{T \times d_{\text{model}}}$$

Layer Norm is applied **before** attention (pre-LN), unlike the original Transformer which uses post-LN. The residual connection preserves the shape.


---

### Equation 3 — MLP sub-layer

$$z_\ell = \text{MLP}\!\left(\text{LN}(z'_\ell)\right) + z'_\ell \quad \in \mathbb{R}^{T \times d_{\text{model}}}$$

The MLP is applied **token-wise**: Linear($d_{\text{model}}$, $d_{\text{ff}}$) → GELU → Linear($d_{\text{ff}}$, $d_{\text{model}}$).
The shape is preserved through every block.

---

### Equation 4 — Classification head

Only the CLS token (row 0 of $z_L$) is used for classification:

$$y = \text{LN}(z_L^0) \quad \in \mathbb{R}^{d_{\text{model}}}$$

$$\text{logits} = \text{Linear}(d_{\text{model}},\, 10) \quad \in \mathbb{R}^{10}$$

The remaining $N$ rows of $z_L$ (patch tokens) are discarded at classification time.




