"""
Differentiable JPEG compression layer using Diff-JPEG (WACV 2024).

This is the state-of-the-art differentiable JPEG implementation that
best approximates real JPEG compression.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from diff_jpeg import DiffJPEGCoding


class DifferentiableJPEG(nn.Module):
    """
    Differentiable JPEG compression layer.

    Uses the Diff-JPEG implementation from WACV 2024 which provides
    the best approximation of real JPEG compression among available
    differentiable implementations.
    """

    def __init__(
        self,
        quality: Union[float, torch.Tensor] = 75.0, # other parameters to JPEG?
        use_ste: bool = True,
        differentiable: bool = True
    ):
        """
        Initialize differentiable JPEG layer.

        Args:
            quality: JPEG quality factor (0-100). Can be a tensor for batched qualities.
            use_ste: Whether to use straight-through estimator for discretizations
            differentiable: Whether to use differentiable operations
        """
        super().__init__()
        self.differentiable = differentiable
        self.use_ste = use_ste

        # Initialize Diff-JPEG module
        self.jpeg_module = DiffJPEGCoding(ste=use_ste)

        # Store quality as buffer (not a parameter, but moves with model)
        if isinstance(quality, (int, float)):
            quality = torch.tensor([quality], dtype=torch.float32)
        self.register_buffer('quality', quality)

    def forward(
        self,
        x: torch.Tensor,
        quality: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply differentiable JPEG compression.

        Args:
            x: Input images in range [0, 255], shape (B, 3, H, W)
            quality: Optional quality override, shape (B,) or scalar

        Returns:
            Compressed images in range [0, 255]
        """
        if quality is None:
            quality = self.quality

        # Ensure quality is the right shape
        if quality.dim() == 0:
            quality = quality.unsqueeze(0)

        # Expand quality to batch size if needed
        if quality.shape[0] == 1 and x.shape[0] > 1:
            quality = quality.expand(x.shape[0])

        # Apply differentiable JPEG
        if self.differentiable:
            compressed = self.jpeg_module(
                image_rgb=x,
                jpeg_quality=quality
            )
        else:
            # Use non-differentiable compression (for evaluation)
            with torch.no_grad():
                compressed = self.jpeg_module(
                    image_rgb=x,
                    jpeg_quality=quality
                )

        return compressed

    def set_quality(self, quality: Union[float, torch.Tensor]):
        """Update the default quality factor."""
        if isinstance(quality, (int, float)):
            quality = torch.tensor([quality], dtype=torch.float32)
        self.quality = quality.to(self.quality.device)


class ExpectationOverTransformation:
    """
    Implements Expectation over Transformation (EoT) for robust adversarial examples.

    As described in "Synthesizing Robust Adversarial Examples" (Athalye et al., 2018),
    this averages gradients over multiple random transformations to create examples
    that are robust to the transformation distribution.

    For JPEG compression, we sample multiple quality factors and average the gradients.
    """

    def __init__(
        self,
        jpeg_layer: DifferentiableJPEG,
        quality_min: float = 50.0,
        quality_max: float = 95.0,
        num_samples: int = 5
    ):
        """
        Initialize EoT for JPEG compression.

        Args:
            jpeg_layer: Differentiable JPEG layer
            quality_min: Minimum quality factor to sample
            quality_max: Maximum quality factor to sample
            num_samples: Number of quality factors to sample per step
        """
        self.jpeg_layer = jpeg_layer
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.num_samples = num_samples

    def sample_qualities(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Sample random JPEG quality factors.

        Args:
            batch_size: Number of quality factors to sample
            device: Device to create tensor on

        Returns:
            Sampled quality factors, shape (batch_size,)
        """
        qualities = torch.rand(batch_size, device=device)
        qualities = qualities * (self.quality_max - self.quality_min) + self.quality_min
        return qualities

    def __call__(
        self,
        x: torch.Tensor,
        single_sample: bool = False
    ) -> torch.Tensor:
        """
        Apply JPEG compression with EoT.

        Args:
            x: Input images, shape (B, 3, H, W)
            single_sample: If True, use a single random quality factor.
                          If False, average over multiple samples.

        Returns:
            Compressed images (averaged over transformations if single_sample=False)
        """
        if single_sample:
            # Single random transformation
            quality = self.sample_qualities(x.shape[0], x.device)
            return self.jpeg_layer(x, quality)
        else:
            # Average over multiple transformations
            compressed_samples = []
            for _ in range(self.num_samples):
                quality = self.sample_qualities(x.shape[0], x.device)
                compressed = self.jpeg_layer(x, quality)
                compressed_samples.append(compressed)

            # Average the compressed outputs
            return torch.stack(compressed_samples).mean(dim=0)
