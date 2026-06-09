### Core idea

Pre-train a ViT in a self-supervised way by masking 75% of the input patches and training the model to reconstruct the missing pixels.
### Architecture

**Asymmetric encoder-decoder :**

- **Encoder** (heavy) : standard ViT, operates only on the **visible patches** (25%)- no mask tokens fed to the encoder
- **Decoder** (lightweight) : takes the latent representations of visible patches learnable **mask tokens** for the missing patches → reconstructs the original pixels

This asymmetry makes training efficient : the encoder processes 4× fewer tokens
→ **3× training speedup** compared to a standard ViT.

### Key design choices

- **Masking ratio : 75%** - high enough to make the task non-trivial (the model cannot just interpolate from neighboring patches)
- **Pixel reconstruction** - the decoder predicts raw pixel values of masked patches

## Latent representation

The encoder compresses each visible patch into a vector $\in \mathbb{R}^{d_{\text{model}}}$.

This vector is called a **latent representation** : it is never directly observed, it is an abstract, compressed summary of what the encoder understood about the patch.
It encodes high-level information : shapes, textures, spatial structure.
This is fundamentally different from raw RGB pixels which carry no abstract meaning.

### MAE vs BERT : the decoder design problem

Both BERT and MAE follow the same principle : mask tokens and predict them.
The key difference is **what the decoder must predict**.

**What is masked :**  
BERT masks words in a sentence. MAE masks patches in an image.

**What the decoder predicts :**  
BERT predicts the missing word. MAE predicts the missing pixels.

**Output space :**  
In BERT the output is discrete : the model picks one word among a vocabulary of 30 000. In MAE the output is continuous : the model predicts 48 raw RGB values per patch.

**Semantic level :**  
In BERT predicting the missing word requires understanding the meaning of the full sentence -there is no shortcut. In MAE predicting missing pixels is a low-level task - the model can cheat by interpolating colors from neighboring patches without understanding the image content.

**Decoder architecture :**  
Because the task is already hard, BERT only needs a simple MLP as decoder : the encoder does all the work. In MAE a lightweight Transformer is needed : a simple MLP is too weak to reconstruct pixels correctly, but a powerful Transformer would cheat by local interpolation, making the encoder lazy and producing poor representations.


