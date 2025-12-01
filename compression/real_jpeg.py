"""
Real JPEG compression for evaluation.

Uses PIL to perform actual JPEG compression (not differentiable),
allowing us to test whether adversarial examples generated with
differentiable JPEG actually transfer to real compression.
"""

import torch
import numpy as np
from PIL import Image
from io import BytesIO

from compression.diff_jpeg import DifferentiableJPEG
import torch.nn.functional as F


class RealJPEGCompression:
    """
    Real JPEG compression using PIL.

    This is used for evaluation to test whether adversarial examples
    generated with differentiable JPEG actually work with real JPEG encoders.
    """

    def __init__(self):
        """Initialize real JPEG compression."""
        pass

    def compress(
        self,
        images: torch.Tensor,
        quality: int
    ) -> torch.Tensor:
        """
        Compress images using real JPEG.

        Args:
            images: Input images, shape (B, 3, H, W) in range [0, 255]
            quality: JPEG quality factor (0-100)

        Returns:
            Compressed images in the same format as input
        """
        device = images.device
        batch_size = images.shape[0]

        # Process each image in the batch
        compressed_list = []
        for i in range(batch_size):
            img = images[i]
            compressed = self._compress_single(img, quality)
            compressed_list.append(compressed)

        # Stack back into batch
        compressed_batch = torch.stack(compressed_list).to(device)
        return compressed_batch

    def _compress_single(
        self,
        image: torch.Tensor,
        quality: int
    ) -> torch.Tensor:
        """Compress a single image."""
        # Convert to numpy and PIL format
        img_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Compress to JPEG in memory
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Load back from JPEG
        img_compressed_pil = Image.open(buffer)
        img_compressed_np = np.array(img_compressed_pil)

        # Convert back to torch tensor
        img_compressed = torch.from_numpy(img_compressed_np).float()
        img_compressed = img_compressed.permute(2, 0, 1)  # (C, H, W)

        return img_compressed

    def compress_to_bytes(
        self,
        image: torch.Tensor,
        quality: int
    ) -> bytes:
        """
        Compress image to JPEG bytes.

        Useful for saving or analyzing file sizes.

        Args:
            image: Single image, shape (3, H, W)
            quality: JPEG quality factor

        Returns:
            JPEG bytes
        """
        # Convert to PIL
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Save to bytes
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        jpeg_bytes = buffer.getvalue()

        return jpeg_bytes
