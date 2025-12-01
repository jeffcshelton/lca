"""
Simple example demonstrating compression-activated adversarial attack generation.

This script shows the minimal code needed to:
1. Load a pretrained model
2. Generate compression-activated adversarial examples
3. Evaluate their effectiveness

Run:
    python examples/simple_attack_example.py
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from torchvision.utils import save_image
from model_factory import ModelFactory
from compression_pgd import CompressionPGD
from compression.real_jpeg import RealJPEGCompression


def load_test_image(image_path: str, size: int = 224) -> torch.Tensor:
    """Load and preprocess a single image."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)  # Scale to [0, 255]
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_image(model: torch.nn.Module, image: torch.Tensor, device: str) -> tuple:
    """Get model prediction for an image."""
    # Normalize for model
    image_norm = image / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    image_norm = (image_norm - mean) / std

    # Get prediction
    with torch.no_grad():
        logits = model(image_norm)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        pred_conf = probs[0, pred_class].item()

    return pred_class, pred_conf


def visualize_results(
    original: torch.Tensor,
    compressed_orig: torch.Tensor,
    adversarial: torch.Tensor,
    compressed_adv: torch.Tensor,
    save_path: str = 'attack_visualization.png'
):
    """Visualize the attack results."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Convert tensors to numpy for visualization
    def tensor_to_numpy(t):
        return t.squeeze(0).cpu().numpy().transpose(1, 2, 0) / 255.0

    axes[0, 0].imshow(tensor_to_numpy(original))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(tensor_to_numpy(compressed_orig))
    axes[0, 1].set_title('Original (Compressed)')
    axes[0, 1].axis('off')

    axes[0, 2].remove()

    # Adversarial image (pre-compression)
    axes[1, 0].imshow(tensor_to_numpy(adversarial))
    axes[1, 0].set_title('Adversarial (Pre-compression)')
    axes[1, 0].axis('off')

    # Compressed adversarial
    axes[1, 1].imshow(tensor_to_numpy(compressed_adv))
    axes[1, 1].set_title('Adversarial (Post-compression)')
    axes[1, 1].axis('off')

    # Perturbation (amplified 10x for visibility)
    perturbation = (adversarial - original).abs() * 10
    axes[1, 2].imshow(tensor_to_numpy(perturbation), cmap='hot')
    axes[1, 2].set_title('Perturbation (10x magnified)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', required=True)
    parser.add_argument('-l', '--label', required=True, type=int)
    parser.add_argument('-o', '--output', default='adv.png')
    parser.add_argument('-t', '--target', type=int)
    parser.add_argument('-e', '--epsilon', type=int, default=8)
    parser.add_argument('-a', '--alpha', type=int, default=2)
    parser.add_argument('-n', '--steps', type=int, default=50)
    parser.add_argument('-d', '--norm', default='linf')
    parser.add_argument('-q', '--quality', default='50')
    parser.add_argument('-s', '--samples', type=int, default=1)
    parser.add_argument('-v', '--visualization', default='vis.png')

    args = parser.parse_args()
    quality = [int(q) for q in args.quality.split('-')]

    if len(quality) == 1:
        quality_min = quality_max = quality[0]
    elif len(quality) == 2:
        quality_min, quality_max = quality
    else:
      raise ValueError('Quality argument invalid')

    print("="*80)
    print("Compression-Activated Adversarial Attack - Simple Example")
    print("="*80 + "\n")

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 1. Load pretrained model
    print("Step 1: Loading pretrained ResNet-50...")
    model = ModelFactory.create_model(
        architecture='resnet50',
        pretrained=True,
        device=device
    )
    print("✓ Model loaded\n")

    # 2. Load test image
    print("Step 2: Loading test image...")

    image = load_test_image(args.image, size=224).to(device)
    true_label = torch.tensor([int(args.label)], device=device)
    print("✓ Image loaded\n")

    # Get original prediction
    orig_class, orig_conf = predict_image(model, image, device)
    print(f"Original prediction: Class {orig_class} (confidence: {orig_conf:.2%})")

    # 3. Initialize attack
    print("\nStep 3: Initializing compression-PGD attack...")
    attack = CompressionPGD(
        model=model,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.steps,
        norm=args.norm,
        targeted=args.target is not None,
        jpeg_quality_min=quality_min,
        jpeg_quality_max=quality_max,
        num_jpeg_samples=args.samples,
        device=device,
        loss_weights={
            'pre_compression': 0.5,
            'post_compression': 1.0,
            'lpips': 0,
            'tv_norm': 0
        }
    )
    print("✓ Attack initialized\n")

    # 4. Generate adversarial example
    print("Step 4: Generating adversarial example...")
    print("(This may take a minute...)\n")

    target_label = args.target if args.target else -1
    adv_image, info = attack.attack(
        images=image,
        labels=true_label,
        return_history=True,
        target_labels=torch.tensor([target_label], device=device)
    )

    print("✓ Attack completed!\n")

    # 5. Evaluate results
    print("="*80)
    print("RESULTS")
    print("="*80)

    # Pre-compression
    pre_class, pre_conf = predict_image(model, adv_image, device)
    print(f"\n✓ PRE-COMPRESSION (adversarial image):")
    print(f"  Predicted class: {pre_class}")
    print(f"  Confidence: {pre_conf:.2%}")
    print(f"  Correctly classified: {pre_class == true_label.item()}")

    # Post-compression with real JPEG
    print(f"\n✓ POST-COMPRESSION (real JPEG):")
    real_jpeg = RealJPEGCompression()
    best_misclass = 50

    for quality in [50, 75, 95]:
        compressed = real_jpeg.compress(adv_image, quality)
        post_class, post_conf = predict_image(model, compressed, device)
        misclassified = post_class != true_label.item()

        if misclassified:
            best_misclass = quality

        print(f"\n  Quality {quality}:")
        print(f"    Predicted class: {post_class}")
        print(f"    Confidence: {post_conf:.2%}")
        print(f"    Misclassified: {post_class != true_label.item()}")

    # Perturbation statistics
    print(f"\n✓ PERTURBATION STATISTICS:")
    delta = adv_image - image
    print(f"  L-infinity norm: {delta.abs().max():.2f} / 255")
    print(f"  L2 norm: {delta.pow(2).sum().sqrt():.2f}")
    print(f"  Mean absolute change: {delta.abs().mean():.4f}")

    print("\n" + "="*80)

    # Output the (uncompressed) adversarial image.
    save_image(adv_image.float() / 255.0, args.output)

    # 6. Visualize
    print("\nStep 5: Creating visualization...")
    compressed_orig = real_jpeg.compress(image, best_misclass)
    compressed_adv = real_jpeg.compress(adv_image, best_misclass)
    visualize_results(
        original=image,
        compressed_orig=compressed_orig,
        adversarial=adv_image,
        compressed_adv=compressed_adv,
        save_path=args.visualization,
    )

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
