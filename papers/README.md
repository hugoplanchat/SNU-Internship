# Literature

Bibliography of papers read for the JEPA project. PDFs are not tracked in git — only this index and personal notes are versioned.

---

## Core papers

### LeCun, 2022 — A Path Towards Autonomous Machine Intelligence
**Authors:** Yann LeCun
**Year:** 2022
**Venue:** OpenReview (position paper)
**Link:** https://openreview.net/forum?id=BZ5a1r-kVsf

LeCun's vision paper introducing the JEPA (Joint Embedding Predictive Architecture) framework. Argues for world-model–based learning over generative and contrastive approaches. Covers hierarchical planning, intrinsic motivation, and the H-JEPA architecture.

📝 Notes: `Lecture_Notes/LeCun-2022-A-Path-Towards-Autonomous-Machine-Intelligence/`

---

### LeCun et al., 2006 — A Tutorial on Energy-Based Learning
**Authors:** Yann LeCun, Sumit Chopra, Raia Hadsell, Marc'Aurelio Ranzato, Fu Jie Huang
**Year:** 2006
**Venue:** Predicting Structured Data, MIT Press
**Link:** http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

Foundational reference on Energy-Based Models (EBMs). Defines energy functions, loss functionals, and explains why contrastive methods are needed to shape the energy landscape. Essential background for understanding non-contrastive methods like JEPA.

📝 Notes: `Lecture_Notes/LeCun EBL/`

---

## Architectures

### Vaswani et al., 2017 — Attention is All You Need
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**Year:** 2017
**Venue:** NeurIPS
**Link:** https://arxiv.org/abs/1706.03762

The Transformer paper. Introduces scaled dot-product attention, multi-head attention, and sinusoidal positional encoding. Removes recurrence entirely, enabling parallel training. Foundation of every modern LLM and ViT-based model.

📝 Notes: `Lecture_Notes/Attention is all you need/`

---

### Dosovitskiy et al., 2021 — An Image is Worth 16x16 Words (ViT)
**Authors:** Alexey Dosovitskiy et al.
**Year:** 2021
**Venue:** ICLR
**Link:** https://arxiv.org/abs/2010.11929

Applies the Transformer architecture directly to images by splitting them into fixed-size patches treated as tokens. Shows that pure attention without convolution outperforms CNNs when pretrained at scale. The ViT encoder is the backbone of I-JEPA.

📝 Notes: (to be written)

---

## Applications

### Mine-JEPA — In-Domain Self-Supervised Learning for Side-Scan Sonar
**Authors:** Authors from SNU, KAIST, Yonsei University
**Year:** 2026
**Venue:** arXiv:2604.00383
**Link:** https://arxiv.org/abs/2604.00383

Applies JEPA (with LeJEPA + SIGReg regularization) to mine-like object classification in side-scan sonar. Pretrains on 1 170 in-domain sonar images and outperforms DINOv3. Directly relevant to the SNU internship and the USV perception use case.

📝 Notes: (to be written)
