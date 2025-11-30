# Latent Compression Attacks
Code for creating adversarial examples that lie dormant until JPEG compression is applied.

## Usage Guide

### Single Example

To create and visualize a single adversarial image, run `attack_example.py`.

#### Key Arguments

| Flag | Meaning |
|------|---------|
| `-i, --image` | Path to input image (required) |
| `-l, --label` | Ground-truth ImageNet label (required) |
| `-o, --output` | Output adversarial image (default: `adv.png`) |
| `-t, --target` | Target label (optional, else untargeted) |
| `-e, --epsilon` | Max perturbation budget (default: 8) |
| `-a, --alpha` | Step size (default: 2) |
| `-n, --steps` | PGD steps (default: 50) |
| `-d, --norm` | `linf` or `l2` |
| `-q, --quality` | JPEG quality or range (e.g. `50` or `50-95`) |
| `-s, --samples` | JPEG samples per step (default: 1) |
| `-v, --visualization` | Visualization output (default: `vis.png`) |

### Run Experiments

To run experiments, run `train_attack.py` with a configuration file.

#### Example
`python train_attack.py --config 'configs/basic_attack.yaml'`