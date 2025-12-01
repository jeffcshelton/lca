"""
Main script for generating compression-activated adversarial examples.
"""

import os
import argparse
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS, default_loader
import numpy as np
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_factory import ModelFactory
from compression_pgd import CompressionPGD
from compression.real_jpeg import RealJPEGCompression


class CompressionAttackExperiment:
    """Manages the entire attack generation experiment."""

    def __init__(self, config_path: str):
        """Initialize experiment from config file."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up device
        self.device = torch.device(
            self.config['experiment']['device']
            if torch.cuda.is_available()
            else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Set random seed for reproducibility
        self._set_seed(self.config['experiment']['seed'])

        # Create output directories
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print("Loading model...")
        self.model = self._load_model()

        # Initialize attack
        print("Initializing attack...")
        self.attack = self._init_attack()

        # Initialize data
        print("Loading dataset...")
        self.dataloader = self._load_data()

        # Initialize real JPEG for evaluation
        self.real_jpeg = RealJPEGCompression()

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _load_model(self) -> nn.Module:
        """Load pretrained classification model."""
        model = ModelFactory.create_model(
            architecture=self.config['model']['architecture'],
            pretrained=self.config['model']['pretrained'],
            num_classes=self.config['model']['num_classes'],
            device=self.device
        )
        return model

    def _init_attack(self) -> CompressionPGD:
        """Initialize the attack."""
        attack_config = self.config['attack']

        return CompressionPGD(
            model=self.model,
            epsilon=attack_config['epsilon'],
            alpha=attack_config['alpha'],
            num_steps=attack_config['num_steps'],
            norm=attack_config['norm'],
            targeted=attack_config['target'] is not None,
            jpeg_quality_min=attack_config['jpeg_quality_min'],
            jpeg_quality_max=attack_config['jpeg_quality_max'],
            num_jpeg_samples=attack_config['num_jpeg_samples'],
            loss_weights=attack_config['loss_weights'],
            device=self.device
        )

    def _load_data(self) -> DataLoader:
        """Load dataset."""
        data_config = self.config['data']

        # Define transforms
        transform_list = [
            transforms.Resize(data_config['image_size']),
            transforms.CenterCrop(data_config['image_size']),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0)  # Convert to [0, 255]
        ]

        transform = transforms.Compose(transform_list)

        # Load dataset
        if data_config['dataset'] == 'imagenet':

            def find_classes(directory):
                """Find classes using folder names as class indices."""
                classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
                class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
                return classes, class_to_idx

            dataset = DatasetFolder(
                root=data_config['data_dir'],
                loader=default_loader,
                extensions=IMG_EXTENSIONS,
                transform=transform,
                is_valid_file=None
            )
            # Override the class mapping to use folder names as labels
            dataset.classes, dataset.class_to_idx = find_classes(data_config['data_dir'])
            dataset.samples = datasets.folder.make_dataset(
                data_config['data_dir'],
                dataset.class_to_idx,
                IMG_EXTENSIONS,
                is_valid_file=None
            )
            dataset.targets = [s[1] for s in dataset.samples]
        else:
            raise ValueError(f"Unknown dataset: {data_config['dataset']}")

        # Limit number of samples if specified
        if data_config['num_samples'] > 0:
            indices = np.random.choice(
                len(dataset),
                min(data_config['num_samples'], len(dataset)),
                replace=False
            )
            dataset = torch.utils.data.Subset(dataset, indices)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=self.config['experiment']['num_workers'],
            pin_memory=True
        )

        return loader

    def run(self):
        """Run the attack generation experiment."""
        print("\n" + "="*80)
        print("Starting Compression-Activated Adversarial Attack Generation")
        print("="*80 + "\n")

        all_results = []
        save_count = 0
        max_saves = self.config['evaluation']['num_examples_to_save']
        target_labels = None

        # Create directories for saving examples
        if self.config['evaluation']['save_examples']:
            examples_dir = self.output_dir / 'adversarial_examples'
            examples_dir.mkdir(exist_ok=True)

        # Generate adversarial examples
        for images, labels in tqdm(self.dataloader, desc="Batches"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.config['attack']['target'] is not None:
                target_labels = torch.tensor(
                    [self.config['attack']['target']],
                    device=self.device,
                ).repeat(images.size(0))

            # Generate adversarial examples
            adv_images, info = self.attack.attack(
                images=images,
                labels=labels,
                return_history=False,
                target_labels=target_labels,
            )

            # Evaluate with real JPEG
            real_jpeg_results = self._evaluate_real_jpeg(
                adv_images, labels, prefix='true'
            )
            info.update(real_jpeg_results)

            if target_labels is not None:
                target_results = self._evaluate_real_jpeg(
                  adv_images, target_labels, prefix='target'
                )
                info.update(target_results)

            # Save examples
            if (self.config['evaluation']['save_examples'] and
                save_count < max_saves):

                num_to_save = min(
                    len(images),
                    max_saves - save_count
                )

                for i in range(num_to_save):
                    self._save_example(
                        clean=images[i],
                        adversarial=adv_images[i],
                        label=labels[i].item(),
                        example_id=save_count,
                        save_dir=examples_dir
                    )
                    save_count += 1

            all_results.append(info)

        # Aggregate results
        final_results = self._aggregate_results(all_results)

        # Print results
        self._print_results(final_results)

        # Save results
        self._save_results(final_results)

        print("\nExperiment completed!")

    def _evaluate_real_jpeg(
        self,
        adv_images: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
    ) -> dict:
        """Evaluate adversarial examples with real JPEG compression."""
        results = {}

        for quality in self.config['evaluation']['real_jpeg_qualities']:
            # Compress with real JPEG
            compressed = self.real_jpeg.compress(adv_images, quality)
            compressed = compressed.to(self.device)

            # Compute accuracy
            accuracy = self._compute_accuracy(compressed, labels)
            results[f'{prefix}_q{quality}_accuracy'] = accuracy

        # results['uncompressed_accuracy'] = self._compute_accuracy(adv_images, labels)?
        return results

    def _compute_accuracy(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Compute classification accuracy."""
        # Normalize for model
        images_norm = images / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        images_norm = (images_norm - mean) / std

        with torch.no_grad():
            logits = self.model(images_norm)
            pred = logits.argmax(dim=1)
            accuracy = (pred == labels).float().mean().item()

        return accuracy

    def _save_example(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        label: int,
        example_id: int,
        save_dir: Path
    ):
        """Save an adversarial example with clean image for comparison."""
        import torchvision

        # Convert to [0, 1] range
        clean = clean / 255.0
        adversarial = adversarial / 255.0

        # Save images
        torchvision.utils.save_image(
            clean,
            save_dir / f'example_{example_id:04d}_clean.png'
        )
        torchvision.utils.save_image(
            adversarial,
            save_dir / f'example_{example_id:04d}_adv.png'
        )

        # Save metadata
        metadata = {
            'example_id': example_id,
            'true_label': label
        }

        with open(save_dir / f'example_{example_id:04d}_meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def _aggregate_results(self, all_results: list) -> dict:
        """Aggregate results across all batches."""
        aggregated = {}

        # Get all keys from first result
        if not all_results:
            return aggregated

        keys = all_results[0].keys()

        for key in keys:
            if isinstance(all_results[0][key], dict):
                # Handle nested dicts (e.g., post_compression_accuracy)
                aggregated[key] = {}
                for subkey in all_results[0][key].keys():
                    values = [r[key][subkey] for r in all_results]
                    aggregated[key][subkey] = np.mean(values)
            else:
                # Handle scalar values
                values = [r[key] for r in all_results]
                aggregated[key] = np.mean(values)

        return aggregated

    def _print_results(self, results: dict):
        """Print experiment results."""
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        print(f"\nPre-compression accuracy: {results['pre_compression_accuracy']:.4f}")

        print("\nPost-compression accuracy (differentiable JPEG):")
        for quality, acc in results['post_compression_accuracy'].items():
            print(f"  {quality}: {acc:.4f}")

        print("\nPost-compression accuracy (real JPEG):")
        for key, acc in results.items():
            if 'real_jpeg' in key:
                quality = key.split('_')[2]
                print(f"  {quality}: {acc:.4f}")

        print(f"\nPerturbation norms:")
        print(f"  L-inf: {results['linf_norm']:.4f}")
        print(f"  L2: {results['l2_norm']:.4f}")

        print("\n" + "="*80)

    def _save_results(self, results: dict):
        """Save results to file."""
        results_file = self.output_dir / 'results.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate compression-activated adversarial examples'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Run experiment
    experiment = CompressionAttackExperiment(args.config)
    experiment.run()


if __name__ == '__main__':
    main()
