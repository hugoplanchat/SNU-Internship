# Papers

Bibliography of papers read for the JEPA project. PDFs are not tracked in git — only this index and personal notes are versioned.

---

## Core papers

### LeCun, 2022 — A Path Towards Autonomous Machine Intelligence
**Authors:** Yann LeCun
**Year:** 2022
**Venue:** OpenReview (position paper)
**Link:** https://openreview.net/forum?id=BZ5a1r-kVsf

LeCun's vision paper introducing the JEPA (Joint Embedding Predictive Architecture) framework. Argues for world-model–based learning over generative and contrastive approaches. Covers hierarchical planning, intrinsic motivation, and the H-JEPA architecture.

📝 Notes: `notes/LeCun-2022-A-Path-Towards-Autonomous-Machine-Intelligence/`

---

### LeCun et al., 2006 — A Tutorial on Energy-Based Learning
**Authors:** Yann LeCun, Sumit Chopra, Raia Hadsell, Marc'Aurelio Ranzato, Fu Jie Huang
**Year:** 2006
**Venue:** Predicting Structured Data, MIT Press
**Link:** http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

Foundational reference on Energy-Based Models (EBMs). Defines energy functions, loss functionals, and explains why contrastive methods are needed to shape the energy landscape. Essential background for understanding non-contrastive methods like JEPA.

📝 Notes: `notes/LeCun EBL/`

---

## Architectures

### Krizhevsky et al., 2012 — ImageNet Classification with Deep CNNs (AlexNet)
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
**Year:** 2012
**Venue:** NeurIPS
**Link:** https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

The paper that launched modern deep learning for vision. A deep CNN (ReLU, dropout, GPU training, data augmentation) wins ImageNet by a large margin. Baseline reference for the convolutional backbone in the benchmark.

📝 Notes: `notes/AlexNet/`

---

### Vaswani et al., 2017 — Attention is All You Need
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**Year:** 2017
**Venue:** NeurIPS
**Link:** https://arxiv.org/abs/1706.03762

The Transformer paper. Introduces scaled dot-product attention, multi-head attention, and sinusoidal positional encoding. Removes recurrence entirely, enabling parallel training. Foundation of every modern LLM and ViT-based model.

📝 Notes: `notes/Attention is all you need/`

---

### Dosovitskiy et al., 2021 — An Image is Worth 16x16 Words (ViT)
**Authors:** Alexey Dosovitskiy et al.
**Year:** 2021
**Venue:** ICLR
**Link:** https://arxiv.org/abs/2010.11929

Applies the Transformer architecture directly to images by splitting them into fixed-size patches treated as tokens. Shows that pure attention without convolution outperforms CNNs when pretrained at scale. The ViT encoder is the shared backbone of the benchmark and of I-JEPA.

📝 Notes: `notes/Vit/`

---

## Self-supervised learning methods

These are the methods reproduced in the `benchmark/` on a shared ViT-Tiny backbone, building up towards JEPA.

### van den Oord et al., 2018 — Representation Learning with Contrastive Predictive Coding (CPC)
**Authors:** Aaron van den Oord, Yazhe Li, Oriol Vinyals
**Year:** 2018
**Venue:** arXiv
**Link:** https://arxiv.org/abs/1807.03748

Introduces the **InfoNCE** loss and predictive contrastive learning: an autoregressive context predicts future latent representations, with a log-bilinear critic. Provides the mutual-information lower bound `I ≥ log(N) − L` that underpins all later contrastive methods.

📝 Notes: `notes/Contrastive_Predictive_Coding/`

### He et al., 2022 — Masked Autoencoders Are Scalable Vision Learners (MAE)
**Authors:** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
**Year:** 2022
**Venue:** CVPR
**Link:** https://arxiv.org/abs/2111.06377

Generative (masked-modeling) SSL: mask ~75% of patches and reconstruct them. The encoder only sees the visible 25%, making pre-training cheap. Excellent fine-tuning, weaker linear probe — the signature of reconstruction-based features.

📝 Notes: `notes/MAE/`

### Chen et al., 2020 — A Simple Framework for Contrastive Learning (SimCLR)
**Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
**Year:** 2020
**Venue:** ICML
**Link:** https://arxiv.org/abs/2002.05709

Contrastive SSL with the **NT-Xent** loss (InfoNCE with cosine similarity + temperature). Key findings: composition of augmentations defines the task, a projection head helps, and large batches (more negatives) matter. Negatives come from the current batch only.

📝 Notes: `notes/SIMCLR/`

### He et al., 2020 — Momentum Contrast (MoCo)
**Authors:** Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
**Year:** 2020
**Venue:** CVPR
**Link:** https://arxiv.org/abs/1911.05722

Contrastive learning as dictionary look-up. A **queue** decouples the number of negatives from the batch size, and a **momentum** key encoder keeps the dictionary consistent. Achieves large, consistent dictionaries without huge batches (unlike SimCLR).

📝 Notes: `notes/MoCo/`

### Grill et al., 2020 — Bootstrap Your Own Latent (BYOL)
**Authors:** Jean-Bastien Grill et al. (DeepMind)
**Year:** 2020
**Venue:** NeurIPS
**Link:** https://arxiv.org/abs/2006.07733

Non-contrastive SSL: **no negatives at all**. An online network predicts the representation of a momentum target network of the same image. Avoids collapse via the predictor + stop-gradient + momentum target, challenging the belief that negatives are necessary.

📝 Notes: `notes/BYOL/`

### Caron et al., 2021 — Emerging Properties in Self-Supervised ViTs (DINO)
**Authors:** Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
**Year:** 2021
**Venue:** ICCV
**Link:** https://arxiv.org/abs/2104.14294

Self-distillation with no labels: a student matches a momentum teacher's output distribution, with centering + sharpening to prevent collapse. Produces strong k-NN features and emergent attention maps that segment objects. Designed with ViTs in mind.

📝 Notes: `notes/DINO/`

### Bardes, Ponce, LeCun, 2022 — VICReg
**Authors:** Adrien Bardes, Jean Ponce, Yann LeCun
**Year:** 2022
**Venue:** ICLR
**Link:** https://arxiv.org/abs/2105.04906

Non-contrastive SSL based on an explicit regularization of the embeddings: **Variance** (keep per-dimension variance above a threshold), **Invariance** (match the two views), **Covariance** (decorrelate dimensions). Prevents collapse without negatives, momentum, or stop-gradient. Conceptually close to the JEPA philosophy.

📝 Notes: `notes/VICReg/`

---

## Applications

### Mine-JEPA — In-Domain Self-Supervised Learning for Side-Scan Sonar
**Authors:** Authors from SNU, KAIST, Yonsei University
**Year:** 2026
**Venue:** arXiv:2604.00383
**Link:** https://arxiv.org/abs/2604.00383

Applies JEPA (with LeJEPA + SIGReg regularization) to mine-like object classification in side-scan sonar. Pretrains on 1 170 in-domain sonar images and outperforms DINOv3. Directly relevant to the SNU internship and the USV perception use case.

📝 Notes: (to be written)
