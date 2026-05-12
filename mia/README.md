# mia — Membership Inference Attacks

Implementations adapted from [MIMIR](https://github.com/iamgroot42/mimir) (Duan et al., 2024).

Original code rewritten as standalone functions using PyTorch/HuggingFace directly (no MIMIR dependencies).

## Attacks

| Function | Reference |
|---|---|
| `loss` | Yeom et al. (2018), "Privacy Risk in Machine Learning" |
| `zlib` | Carlini et al. (2021), "Extracting Training Data from Large Language Models" |
| `min_k` | Shi et al. (2023), "Detecting Pretraining Data from Large Language Models" |
| `min_k_plus_plus` | Zhang et al. (2024), github.com/zjysteven/mink-plus-plus |
| `reference` | Carlini et al. (2021) |
| `gradnorm` | Gradient-based MI via MIMIR |
