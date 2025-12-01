"""
Projected Gradient Descent (PGD) attack with differentiable compression layer.

This implements the core attack algorithm for generating compression-activated
adversarial examples.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from compression.diff_jpeg import DifferentiableJPEG, ExpectationOverTransformation
from loss_functions import CompressionActivatedLoss


class CompressionPGD:
    """
    PGD attack with differentiable JPEG compression.

    Generates adversarial examples that are:
    - Correctly classified before compression
    - Misclassified after compression
    - Perceptually similar to the original image
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8.0,
        alpha: float = 2.0,
        num_steps: int = 100,
        norm: str = 'linf',
        targeted: bool = False,
        jpeg_quality_min: float = 50.0,
        jpeg_quality_max: float = 95.0,
        num_jpeg_samples: int = 5,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = 'cuda'
    ):
        """
        Initialize compression-PGD attack.

        Args:
            model: Target classification model
            epsilon: Maximum perturbation size
            alpha: Step size for each iteration
            num_steps: Number of PGD steps
            norm: Perturbation norm ('linf' or 'l2')
            targeted: Whether to perform targeted attack
            jpeg_quality_min: Minimum JPEG quality for EoT
            jpeg_quality_max: Maximum JPEG quality for EoT
            num_jpeg_samples: Number of JPEG samples for EoT
            loss_weights: Weights for loss components
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.norm = norm
        self.targeted = targeted
        self.device = device

        # Initialize JPEG layer and EoT
        self.jpeg_layer = DifferentiableJPEG(use_ste=True).to(device)
        self.eot = ExpectationOverTransformation(
            self.jpeg_layer,
            quality_min=jpeg_quality_min,
            quality_max=jpeg_quality_max,
            num_samples=num_jpeg_samples
        )

        # Initialize loss function
        self.loss_fn = CompressionActivatedLoss(
            model=model,
            targeted=targeted,
            loss_weights=loss_weights,
            device=device
        )

        self.model.eval()

    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        return_history: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate compression-activated adversarial examples.

        Args:
            images: Clean images, shape (B, 3, H, W) in range [0, 255]
            labels: True labels, shape (B,)
            target_labels: Target labels for targeted attacks, shape (B,)
            return_history: Whether to return optimization history

        Returns:
            Tuple of (adversarial_images, info_dict)
        """
        batch_size = images.shape[0]

        # Initialize adversarial perturbation
        delta = torch.zeros_like(images, requires_grad=True, device=self.device)

        # Optimization history
        history = {
            'losses': [],
            'pre_accuracy': [],
            'post_accuracy': []
        } if return_history else None

        # PGD optimization loop
        for step in tqdm(range(self.num_steps), desc="Generating attack"):
            # Apply perturbation
            adv_images = self._apply_perturbation(images, delta)

            # Apply JPEG compression with EoT
            adv_compressed = self.eot(adv_images, single_sample=False)

            # Compute loss
            losses = self.loss_fn(
                adv_pre=adv_images,
                adv_post=adv_compressed,
                original=images,
                true_label=labels,
                target_label=target_labels
            )

            # Backward pass
            losses['total'].backward()

            # Update perturbation with gradient ascent/descent
            with torch.no_grad():
                # Get gradient
                grad = delta.grad.detach()

                # Normalize gradient based on norm
                if self.norm == 'linf':
                    # Sign gradient for L-infinity
                    delta_update = self.alpha * grad.sign()
                elif self.norm == 'l2':
                    # Normalize by L2 norm for L2 constraint
                    grad_norm = torch.norm(
                        grad.view(batch_size, -1), p=2, dim=1
                    ).view(batch_size, 1, 1, 1)
                    delta_update = self.alpha * grad / (grad_norm + 1e-8)
                else:
                    raise ValueError(f"Unsupported norm: {self.norm}")

                # Update delta
                delta.data = delta.data - delta_update

                # Project to epsilon ball
                delta.data = self._project_perturbation(delta.data, images)

                # Zero gradient for next iteration
                delta.grad.zero_()

            # Record history
            if return_history and step % 10 == 0:
                with torch.no_grad():
                    history['losses'].append(losses['total'].item())

                    # Compute accuracy pre and post compression
                    pre_acc = self._compute_accuracy(adv_images, labels)
                    post_acc = self._compute_accuracy(adv_compressed, labels)

                    history['pre_accuracy'].append(pre_acc)
                    history['post_accuracy'].append(post_acc)

        # Final adversarial images
        with torch.no_grad():
            final_adv = self._apply_perturbation(images, delta)

        # Compute final metrics
        info = self._compute_final_metrics(
            final_adv, images, labels, target_labels
        )

        if return_history:
            info['history'] = history

        return final_adv, info

    def _apply_perturbation(
        self,
        images: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """Apply perturbation and clamp to valid range."""
        adv_images = images + delta
        return torch.clamp(adv_images, 0, 255)

    def _project_perturbation(
        self,
        delta: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Project perturbation to epsilon ball."""
        if self.norm == 'linf':
            # L-infinity ball
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        elif self.norm == 'l2':
            # L2 ball
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)

            # Scale down if norm exceeds epsilon
            scale = torch.clamp(self.epsilon / (delta_norm + 1e-8), max=1.0)
            delta = delta * scale.view(batch_size, 1, 1, 1)

        # Ensure adversarial image stays in valid range
        adv_images = images + delta
        delta = torch.clamp(adv_images, 0, 255) - images

        return delta

    def _compute_accuracy(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute classification accuracy."""
        # Normalize for model
        images_norm = self.loss_fn._normalize_for_model(images)

        with torch.no_grad():
            logits = self.model(images_norm)
            pred = logits.argmax(dim=1)
            accuracy = (pred == labels).float().mean().item()

        return accuracy

    def _compute_final_metrics(
        self,
        adv_images: torch.Tensor,
        clean_images: torch.Tensor,
        true_labels: torch.Tensor,
        target_labels: Optional[torch.Tensor]
    ) -> Dict:
        """Compute comprehensive metrics for the attack."""
        with torch.no_grad():
            # Pre-compression metrics
            pre_acc = self._compute_accuracy(adv_images, true_labels)

            # Post-compression metrics (test with different qualities)
            test_qualities = [50, 75, 85, 95]
            post_accuracies = {}

            for quality in test_qualities:
                self.jpeg_layer.set_quality(float(quality))
                compressed = self.jpeg_layer(adv_images)
                post_acc = self._compute_accuracy(compressed, true_labels)
                post_accuracies[f'q{quality}'] = post_acc

            # Perturbation norms
            delta = adv_images - clean_images
            linf_norm = torch.abs(delta).max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].mean().item()
            l2_norm = torch.norm(
                delta.view(delta.shape[0], -1), p=2, dim=1
            ).mean().item()

        return {
            'pre_compression_accuracy': pre_acc,
            'post_compression_accuracy': post_accuracies,
            'linf_norm': linf_norm,
            'l2_norm': l2_norm
        }
