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

# Import our modules
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
    adversarial: torch.Tensor,
    compressed: torch.Tensor,
    save_path: str = 'attack_visualization.png'
):
    """Visualize the attack results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Convert tensors to numpy for visualization
    def tensor_to_numpy(t):
        return t.squeeze(0).cpu().numpy().transpose(1, 2, 0) / 255.0

    # Original image
    axes[0].imshow(tensor_to_numpy(original))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Adversarial image (pre-compression)
    axes[1].imshow(tensor_to_numpy(adversarial))
    axes[1].set_title('Adversarial (Pre-compression)')
    axes[1].axis('off')

    # Compressed adversarial
    axes[2].imshow(tensor_to_numpy(compressed))
    axes[2].set_title('Adversarial (Post-compression)')
    axes[2].axis('off')

    # Perturbation (amplified 10x for visibility)
    perturbation = (adversarial - original).abs() * 10
    axes[3].imshow(tensor_to_numpy(perturbation), cmap='hot')
    axes[3].set_title('Perturbation (10x magnified)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


def main():
    """Main demonstration."""
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
    # Note: You'll need to provide your own test image
    # For this example, we'll create a random image
    print("Step 2: Loading test image...")
    # image = torch.rand(1, 3, 224, 224, device=device) * 255.0
    image = load_test_image('tabbycat.jpg', size=224).to(device)
    true_label = torch.tensor([281], device=device)  # Example: 'tabby cat'
    print("✓ Image loaded\n")

    # Get original prediction
    orig_class, orig_conf = predict_image(model, image, device)
    print(f"Original prediction: Class {orig_class} (confidence: {orig_conf:.2%})")

    # 3. Initialize attack
    print("\nStep 3: Initializing compression-PGD attack...")
    attack = CompressionPGD(
        model=model,
        epsilon=8.0,          # Max perturbation
        alpha=2.0,            # Step size
        num_steps=50,         # Number of iterations (reduced for demo)
        norm='linf',          # L-infinity norm
        targeted=True,       # Untargeted attack
        jpeg_quality_min=50,  # EoT quality range
        jpeg_quality_max=50,
        num_jpeg_samples=1,   # EoT samples
        device=device,
        loss_weights={
            'pre_compression': 0,
            'post_compression': 1.0,
            'lpips': .5,
            'tv_norm': 0
        }
    )
    print("✓ Attack initialized\n")

    # 4. Generate adversarial example
    print("Step 4: Generating adversarial example...")
    print("(This may take a minute...)\n")

    adv_image, info = attack.attack(
        images=image,
        labels=true_label,
        return_history=True,
        target_labels=torch.tensor([7], device=device)
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

    for quality in [50, 75, 95]:
        compressed = real_jpeg.compress(adv_image, quality)
        post_class, post_conf = predict_image(model, compressed, device)

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

    # 6. Visualize
    print("\nStep 5: Creating visualization...")
    compressed_q75 = real_jpeg.compress(adv_image, 50)
    visualize_results(
        original=image,
        adversarial=adv_image,
        compressed=compressed_q75
    )

    print("\n✓ Done!")


if __name__ == '__main__':
    main()