
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

![[Pasted image 20260609192350.png]]