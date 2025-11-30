# Latent Compression Attacks
Code for creating adversarial examples that lie dormant until JPEG compression is applied.

# Usage Guide

## Single Example

To create and visualize a single adversarial image, run `attack_example.py`.

### Key Arguments

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

### Example

#### Command
`python attack_example.py -i 'goldfish.png' -l 1 -d 'linf' -q '50-95' -s 5`

#### Terminal Output
```
================================================================================
Compression-Activated Adversarial Attack - Simple Example
================================================================================

Using device: cpu

Step 1: Loading pretrained ResNet-50...
✓ Model loaded

Step 2: Loading test image...
✓ Image loaded

Original prediction: Class 1 (confidence: 56.84%)

Step 3: Initializing compression-PGD attack...
✓ Attack initialized

Step 4: Generating adversarial example...
(This may take a minute...)

Generating attack:
Generating attack: 100%|███████████████████████████| 50/50 [00:26<00:00,  1.88it/s]
✓ Attack completed!

================================================================================
RESULTS
================================================================================

✓ PRE-COMPRESSION (adversarial image):
  Predicted class: 1
  Confidence: 90.71%
  Correctly classified: True

✓ POST-COMPRESSION (real JPEG):

  Quality 50:
    Predicted class: 380
    Confidence: 25.01%
    Misclassified: True

  Quality 75:
    Predicted class: 382
    Confidence: 56.40%
    Misclassified: True

  Quality 95:
    Predicted class: 1
    Confidence: 34.91%
    Misclassified: False

✓ PERTURBATION STATISTICS:
  L-infinity norm: 8.00 / 255
  L2 norm: 2080.09
  Mean absolute change: 4.6598

================================================================================

Step 5: Creating visualization...

Visualization saved to: vis.png

✓ Done!
```

#### Output Image
![Sample `vis.png` comparing original to adversarial images](vis.png)


## Run Experiments

To run experiments, run `train_attack.py` with a configuration file.

### Example
`python train_attack.py --config 'configs/basic_attack.yaml'`