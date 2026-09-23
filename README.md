# T-ORR

Code for **T-ORR: Text-Anchored Orthogonal Residual Rectification for Robust Multimodal Sarcasm Detection**.

The model uses text as the semantic anchor. Audio and video are aligned to text tokens, decomposed into resonance and orthogonal dissonance, and then routed through literal and incongruous states for classification.

## Overview

1. **Dynamic Locality-Constrained Alignment** predicts a token-specific temporal center and applies a Gaussian locality mask to audio and video attention.
2. **Structure-Preserving Geometric Decomposition** generates a text-conditioned orthonormal basis with QR decomposition. Projected non-verbal features are split into semantic resonance and orthogonal dissonance.
3. **Contrastive Incongruity Routing (CIR)** builds literal and incongruous states and classifies the concatenation of both states and their absolute semantic distance.
4. The training objective combines task loss, topology preservation in the resonance space, and gate sparsity for sincere samples.

Implementation details and tensor conventions are documented in [docs/implementation.md](docs/implementation.md).

## Repository layout

```text
.
├── configs/default.yaml          # Paper-aligned training defaults
├── scripts/make_synthetic_data.py
├── src/torr/
│   ├── alignment.py              # Equations 1-3
│   ├── decomposition.py          # Equations 4-9 and L_struct
│   ├── routing.py                # Equations 10-14 and L_gate_sparsity
│   ├── model.py                  # End-to-end T-ORR fusion model
│   ├── data.py                   # Variable-length feature batches
│   ├── train.py                  # Training and early stopping
│   └── evaluate.py               # Checkpoint evaluation
└── tests/test_model.py           # Forward, shape, loss, and gradient tests
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data format

The model consumes features from frozen modality encoders:

- text: DeBERTa-v3-Large
- audio: WavLM
- video: DINOv2-Large

Each split is a file produced by `torch.save`. It contains a list of dictionaries:

```python
{
    "id": "utterance-id",
    "text": FloatTensor[L, D_text],
    "audio": FloatTensor[T_audio, D_audio],
    "video": FloatTensor[T_video, D_video],
    "label": 0 or 1,
}
```

Set the input dimensions in `configs/default.yaml` to match the stored embeddings. The collator pads variable-length sequences and constructs masks automatically.

MUStARD and MUStARD++ are not redistributed in this repository. Obtain them from their original authors and follow their respective licenses. Use a strict speaker-independent split: test speakers must not appear in training.

## Quick check

Create synthetic feature files and run one training job:

```bash
python scripts/make_synthetic_data.py --output data --dims 16 16 16 --count 24
python -m torr.train --config configs/smoke.yaml
python -m torr.evaluate --checkpoint checkpoints/smoke.pt --data data/test.pt
```

This only checks the training and evaluation path. It is not a reproduction experiment.

## Training settings

The default configuration uses AdamW, batch size 32, learning rate `1e-4`, weight decay `1e-4`, at most 30 epochs, and early stopping after five epochs without validation-loss improvement. The structure and gate-sparsity coefficients are `0.5` and `0.1`.

For the paper experiments, raw audio is resampled to 16 kHz and represented at 10 ms frame resolution; video is sampled at 15 FPS; text sequence length is capped at 50. Foundational encoders remain frozen so the comparison isolates the trainable fusion strategy.

## Results

The training and evaluation commands report accuracy, precision, recall, and F1. Use the same speaker-independent splits and preprocessing for fair comparison with the paper.

Reported paper results:

| Model | MUStARD P | MUStARD R | MUStARD F1 | MUStARD++ F1 |
|---|---:|---:|---:|---:|
| Simple-Concat | 71.2 | 70.5 | 70.8 | 66.1 |
| DIP | 74.8 | 73.9 | 74.3 | 69.1 |
| **T-ORR** | **77.5** | **76.2** | **76.8** | **72.4** |

Simple-Concat uses the same frozen backbones; the gain therefore targets the fusion and rectification strategy rather than encoder capacity.

## Current scope

Included:

- the T-ORR fusion architecture and all three loss terms;
- variable-length feature batching;
- training, validation, early stopping, and checkpoint evaluation;
- component tests and an end-to-end smoke configuration.

Not included:

- MUStARD or MUStARD++ data;
- pretrained encoder checkpoints;
- dataset split files or extracted features.

These files should be added only when their redistribution terms permit it.

## Citation

```bibtex
@inproceedings{chen2026torr,
  title={T-ORR: Text-Anchored Orthogonal Residual Rectification for Robust Multimodal Sarcasm Detection},
  author={Chen, Qi and Chen, Junyi and Han, Jiabin and Yang, Hongjiao},
  year={2026}
}
```

Please replace the provisional venue fields when the final proceedings record is available.

## License

No open-source license has been selected. Keep the repository private until the authors choose one.

## Contact

For questions about the paper or implementation, contact `chenqi@tjfsu.edu.cn`.
