# Semaine 1 — Liens & fichiers à télécharger

## Lectures à télécharger (PDF)

### LeCun 2022 — Manifeste principal *(Lundi)*
- **Titre :** *A Path Towards Autonomous Machine Intelligence*
- **Lien arXiv :** https://arxiv.org/abs/2206.04695
- **PDF direct :** https://arxiv.org/pdf/2206.04695
- **Taille :** ~65 pages
- > Lis-le **2 fois** dans la semaine (lundi matin + lundi après-midi annoté)

### LeCun 2006 — Energy-Based Learning *(Mardi)*
- **Titre :** *A Tutorial on Energy-Based Learning*
- **PDF direct :** http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
- **Sections à lire :** 1 à 4 uniquement (le reste est optionnel)

### Vaswani et al. 2017 — Attention is All You Need *(Jeudi)*
- **Titre :** *Attention Is All You Need*
- **Lien arXiv :** https://arxiv.org/abs/1706.03762
- **PDF direct :** https://arxiv.org/pdf/1706.03762
- **Section à lire :** Section 3 (les équations d'attention), pas besoin de tout lire

---

## Articles de blog à lire en ligne (pas besoin de télécharger)

### Blog Rohit Bandaru — Deep Dive JEPA *(Mardi)*
- **Chercher sur Medium :** "Deep Dive into Yann LeCun's JEPA Rohit Bandaru"
- **Lien Medium (peut nécessiter un compte) :** https://medium.com/@rohitbandaru/deep-dive-into-yann-lecuns-jepa-b79e4b60e1bf
- > Meilleure intro vulgarisée à I-JEPA, à lire après le tutoriel EBL

### The Illustrated Transformer — Jay Alammar *(Jeudi)*
- **Lien :** https://jalammar.github.io/illustrated-transformer/
- > Référence visuelle pour comprendre Q/K/V avant de coder. Lire avec un crayon.

---

## Tutoriels PyTorch *(Mercredi)*

### PyTorch 60-Minute Blitz
- **Lien :** https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- Couvre : tenseurs, autograd, réseau simple, entraînement

### PyTorch Learn the Basics
- **Lien :** https://pytorch.org/tutorials/beginner/basics/intro.html
- Couvre : Dataset, DataLoader, modèle, loss, optimizer, boucle d'entraînement
- Faire les **5 premières sections** (pas besoin de tout finir)

---

## Talk YouTube LeCun *(Lundi)*

Choisir **un seul** parmi les deux :

### Option A — NeurIPS 2022 (recommandé, ~1h)
- **Chercher sur YouTube :** "Yann LeCun NeurIPS 2022 Path Autonomous Machine Intelligence"
- **Lien YouTube :** https://www.youtube.com/watch?v=OKkEdTchsiE

### Option B — Collège de France (~1h30, plus détaillé)
- **Chercher sur YouTube :** "Yann LeCun Collège de France deep learning"
- **Chaine Collège de France :** https://www.college-de-france.fr/fr/agenda/cours/deep-learning-and-other-machine-learning-methods

---

## Dataset pour le notebook MLP *(Mercredi)*

Utiliser **l'un des deux** (PyTorch les télécharge automatiquement) :

```python
# MNIST (plus simple, recommandé pour commencer)
from torchvision import datasets
dataset = datasets.MNIST(root='./data', train=True, download=True)

# CIFAR-10 (si tu veux un peu plus de challenge)
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
```

Pas de téléchargement manuel nécessaire — PyTorch gère automatiquement.

---

## Récap fichiers à avoir avant lundi matin

| Fichier | Source | Obligatoire |
|---|---|---|
| `lecun-2022-autonomous-machine-intelligence.pdf` | arXiv | ✅ Oui |
| `lecun-06-energy-based-learning.pdf` | yann.lecun.com | ✅ Oui |
| `attention-is-all-you-need.pdf` | arXiv | ✅ Oui (pour jeudi) |
| Talk LeCun YouTube | YouTube | ✅ Oui (lundi) |
| Blog Rohit Bandaru | Medium | ✅ Oui (mardi) |
| The Illustrated Transformer | jalammar.github.io | ✅ Oui (jeudi) |
