"""
Model factory for loading pretrained classification models.

Follows the pattern used in robustness evaluation papers (e.g., RobustBench).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class ModelFactory:
    """Factory for creating pretrained models."""

    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'inception_v3': models.inception_v3,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'efficientnet_b0': models.efficientnet_b0,
    }

    @staticmethod
    def create_model(
        architecture: str,
        pretrained: bool = True,
        num_classes: int = 1000,
        device: str = 'cuda'
    ) -> nn.Module:
        """
        Create a pretrained classification model.

        Args:
            architecture: Model architecture name
            pretrained: Whether to load pretrained weights
            num_classes: Number of output classes
            device: Device to load model on

        Returns:
            Pretrained model in evaluation mode
        """
        if architecture not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {list(ModelFactory.SUPPORTED_MODELS.keys())}"
            )

        # Load model
        weights = 'DEFAULT' if pretrained else None
        model = ModelFactory.SUPPORTED_MODELS[architecture](weights=weights)

        # Modify final layer if needed (for custom datasets)
        if num_classes != 1000:
            model = ModelFactory._modify_classifier(model, architecture, num_classes)

        model = model.to(device)
        model.eval()

        return model

    @staticmethod
    def _modify_classifier(
        model: nn.Module,
        architecture: str,
        num_classes: int
    ) -> nn.Module:
        """Modify the classifier layer for custom number of classes."""
        if 'resnet' in architecture or 'densenet' in architecture:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif 'vgg' in architecture:
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
        elif 'inception' in architecture:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif 'efficientnet' in architecture:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError(
                f"Classifier modification not implemented for {architecture}"
            )

        return model

    @staticmethod
    def get_input_size(architecture: str) -> Tuple[int, int]:
        """Get the expected input size for a model architecture."""
        # Most models use 224x224, except Inception which uses 299x299
        if 'inception' in architecture:
            return (299, 299)
        return (224, 224)

    @staticmethod
    def get_normalization_stats(dataset: str = 'imagenet') -> Tuple[list, list]:
        """
        Get normalization statistics for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Tuple of (mean, std) for normalization
        """
        if dataset == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif dataset == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif dataset == 'cifar100':
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        return mean, std


def test_model_factory():
    """Test the model factory."""
    print("Testing ModelFactory...")

    # Test creating different models
    for arch in ['resnet50', 'vgg16', 'inception_v3']:
        print(f"\nCreating {arch}...")
        model = ModelFactory.create_model(arch, pretrained=True, device='cpu')

        # Test forward pass
        input_size = ModelFactory.get_input_size(arch)
        x = torch.randn(1, 3, *input_size)

        with torch.no_grad():
            output = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (1, 1000), f"Expected (1, 1000), got {output.shape}"

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_model_factory()