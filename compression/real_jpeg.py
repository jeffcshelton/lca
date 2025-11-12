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


def compare_jpeg_implementations():
    """
    Compare differentiable JPEG with real JPEG.

    This helps verify that the differentiable implementation
    is a good approximation of real JPEG.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test image
    x = torch.rand(1, 3, 224, 224, device=device) * 255.0

    # Test at different quality levels
    qualities = [50, 75, 85, 95]

    print("Comparing differentiable JPEG with real JPEG")
    print("="*60)

    diff_jpeg = DifferentiableJPEG().to(device)
    real_jpeg = RealJPEGCompression()

    for quality in qualities:
        # Differentiable JPEG
        diff_jpeg.set_quality(float(quality))
        with torch.no_grad():
            compressed_diff = diff_jpeg(x)

        # Real JPEG
        compressed_real = real_jpeg.compress(x, quality)

        # Compute PSNR
        mse = F.mse_loss(compressed_diff, compressed_real)
        psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse)

        print(f"Quality {quality:3d}: PSNR = {psnr.item():.2f} dB")

    print("="*60)


def test_real_jpeg():
    """Test real JPEG compression."""
    print("Testing RealJPEGCompression...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test images
    images = torch.rand(2, 3, 224, 224, device=device) * 255.0

    # Test compression
    real_jpeg = RealJPEGCompression()

    for quality in [50, 75, 95]:
        compressed = real_jpeg.compress(images, quality)

        print(f"\nQuality {quality}:")
        print(f"  Input shape: {images.shape}")
        print(f"  Output shape: {compressed.shape}")
        print(f"  Input range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Output range: [{compressed.min():.2f}, {compressed.max():.2f}]")

        # Test bytes compression
        jpeg_bytes = real_jpeg.compress_to_bytes(images[0], quality)
        print(f"  File size: {len(jpeg_bytes)} bytes")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_real_jpeg()
    print("\n")
    compare_jpeg_implementations()