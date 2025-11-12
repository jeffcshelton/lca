"""
Custom loss functions for compression-activated adversarial attacks.

Implements the multi-component loss function that:
1. Minimizes loss pre-compression (keep image benign)
2. Maximizes loss post-compression (make image adversarial after compression)
3. Maintains perceptual similarity (LPIPS/SSIM)
4. Reduces artifacts (TV norm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim, ms_ssim
from typing import Dict, Optional


class CompressionActivatedLoss(nn.Module):
    """
    Loss function for compression-activated adversarial attacks.

    This loss encourages images to be:
    - Correctly classified before compression (benign)
    - Misclassified after compression (adversarial)
    - Perceptually similar to the original
    - Free from obvious artifacts
    """

    def __init__(
        self,
        model: nn.Module,
        targeted: bool = False,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = 'cuda'
    ):
        """
        Initialize loss function.

        Args:
            model: Target classification model
            targeted: Whether this is a targeted attack
            loss_weights: Dictionary of loss component weights
            device: Device to run computations on
        """
        super().__init__()
        self.model = model
        self.targeted = targeted
        self.device = device

        # Default loss weights
        default_weights = {
            'pre_compression': 1.0,
            'post_compression': 1.0,
            'lpips': 0.1,
            'tv_norm': 0.01
        }
        self.loss_weights = loss_weights if loss_weights else default_weights

        # Initialize LPIPS metric for perceptual similarity (use SSIM instead?)
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device) # alex?
        self.lpips_fn.eval()

        # Freeze LPIPS weights
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def forward(
        self,
        adv_pre: torch.Tensor,
        adv_post: torch.Tensor,
        original: torch.Tensor,
        true_label: torch.Tensor,
        target_label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss and its components.

        Args:
            adv_pre: Adversarial images before compression, shape (B, 3, H, W)
            adv_post: Adversarial images after compression, shape (B, 3, H, W)
            original: Original clean images, shape (B, 3, H, W)
            true_label: True labels, shape (B,)
            target_label: Target labels for targeted attacks, shape (B,)

        Returns:
            Dictionary containing total loss and individual components
        """
        # Get model predictions
        logits_pre = self.model(self._normalize_for_model(adv_pre))
        logits_post = self.model(self._normalize_for_model(adv_post))

        # 1. Pre-compression loss (minimize to keep benign)
        loss_pre = self._classification_loss(
            logits_pre, true_label, target_label, maximize=False
        )

        # 2. Post-compression loss (maximize to make adversarial)
        loss_post = self._classification_loss(
            logits_post, true_label, target_label, maximize=True
        )

        # 3. Perceptual similarity (LPIPS)
        loss_lpips = self._compute_lpips(adv_pre, original)

        # 4. Total variation for smoothness
        loss_tv = self._total_variation(adv_pre)

        # Combine losses
        total_loss = (
            self.loss_weights['pre_compression'] * loss_pre +
            self.loss_weights['post_compression'] * loss_post +
            self.loss_weights['lpips'] * loss_lpips +
            self.loss_weights['tv_norm'] * loss_tv
        )

        # Return all components for logging
        return {
            'total': total_loss,
            'pre_compression': loss_pre,
            'post_compression': loss_post,
            'lpips': loss_lpips,
            'tv_norm': loss_tv
        }

    def _classification_loss(
        self,
        logits: torch.Tensor,
        true_label: torch.Tensor,
        target_label: Optional[torch.Tensor],
        maximize: bool
    ) -> torch.Tensor:
        """
        Compute classification loss (cross-entropy).

        Args:
            logits: Model logits
            true_label: True class labels
            target_label: Target class labels (for targeted attacks)
            maximize: If True, maximize loss (for adversarial). If False, minimize.
        """
        if self.targeted and target_label is not None:
            # Targeted: minimize loss w.r.t. target class
            loss = F.cross_entropy(logits, target_label)
            return -loss if maximize else loss
        else:
            # Untargeted: maximize loss w.r.t. true class
            loss = F.cross_entropy(logits, true_label)
            return -loss if maximize else loss

    def _compute_lpips(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual distance.

        Args:
            x, y: Images in range [0, 255]
        """
        # LPIPS expects images in [-1, 1]
        x_norm = (x / 255.0) * 2.0 - 1.0
        y_norm = (y / 255.0) * 2.0 - 1.0

        return self.lpips_fn(x_norm, y_norm).mean()

    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation norm for smoothness.

        TV norm penalizes high-frequency artifacts and encourages
        smooth perturbations.

        Args:
            x: Images, shape (B, C, H, W)
        """
        # Differences along height and width
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        return diff_h.mean() + diff_w.mean()

    def _normalize_for_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize images for model input.

        Converts from [0, 255] to ImageNet normalized values.
        """
        # Convert to [0, 1]
        x = x / 255.0

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)

        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)

        return (x - mean) / std


class SSIMLoss(nn.Module):
    """
    SSIM-based perceptual loss.

    Alternative to LPIPS that can be faster to compute.
    """

    def __init__(self, data_range: float = 255.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            x, y: Images in range [0, data_range]
        """
        ssim_val = ssim(x, y, data_range=self.data_range, size_average=True) #ms_ssim?
        return 1.0 - ssim_val


def test_loss_functions():
    """Test the loss functions."""
    print("Testing CompressionActivatedLoss...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 1000, device=x.device)

    model = DummyModel().to(device)

    # Create loss function
    loss_fn = CompressionActivatedLoss(model, device=device)

    # Create dummy data
    batch_size = 2
    adv_pre = torch.rand(batch_size, 3, 224, 224, device=device) * 255
    adv_post = torch.rand(batch_size, 3, 224, 224, device=device) * 255
    original = torch.rand(batch_size, 3, 224, 224, device=device) * 255
    true_label = torch.randint(0, 1000, (batch_size,), device=device)

    # Compute loss
    losses = loss_fn(adv_pre, adv_post, original, true_label)

    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Pre-compression: {losses['pre_compression'].item():.4f}")
    print(f"Post-compression: {losses['post_compression'].item():.4f}")
    print(f"LPIPS: {losses['lpips'].item():.4f}")
    print(f"TV norm: {losses['tv_norm'].item():.4f}")

    # Test backward pass
    losses['total'].backward()
    print("\nBackward pass successful!")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_loss_functions()