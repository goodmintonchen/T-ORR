# Implementation notes

This note records the choices that are easy to miss when reading the model code.

## Tensor layout

All sequence tensors use batch-first layout:

- text: `[B, L, D_t]`
- audio: `[B, T_a, D_a]`
- video: `[B, T_v, D_v]`

The three input projections map encoder-specific dimensions to the common hidden size `D`. Boolean masks use `[B, length]`, with `True` marking valid positions.

## Dynamic alignment

`DynamicLocalityConstrainedAlignment` implements Equations (1)-(3). The shift projection produces one relative offset per text token. The offset is bounded by `max_alignment_shift`, converted to an absolute frame center, and used to build a Gaussian log mask. The mask is added to multi-head attention scores before softmax.

Audio and video use separate parameters because their sampling rates and local timing patterns differ.

## Geometric decomposition

For each text token, `basis_generator` predicts `K` vectors in the hidden space. Reduced QR decomposition turns them into an orthonormal basis `U`. The mapped non-verbal feature is projected as

```text
F_cons = U (U^T E')
F_disc = E' - F_cons
```

The two components reconstruct the mapped feature up to floating-point error. The topology loss compares pairwise cosine-similarity matrices after sequence pooling and constrains only the resonance component.

## Contrastive routing

The literal state adds transformed resonance to the text representation. The incongruous state gates the text representation with dissonance. After masked mean pooling, the classifier receives

```text
[H_lit, H_inc, abs(H_lit - H_inc)]
```

Gate sparsity is applied only to non-sarcastic samples. This prevents ordinary non-verbal variation from activating the incongruous path.

## Encoder boundary

The repository starts from frozen encoder features. Feature extraction is intentionally kept outside the model package so experiments can cache embeddings and compare fusion strategies with identical inputs. A reproduction should record encoder revision, preprocessing version, split definition, and feature dimension alongside every feature archive.

## Numerical checks

The component tests cover:

- masked dynamic alignment and attention normalization;
- reconstruction of mapped features from resonance and dissonance;
- orthonormality of the QR basis;
- finite losses and gradient flow through the classifier.

