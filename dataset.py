import os

from argparse import ArgumentParser
from datasets import load_dataset
from PIL import Image

dataset = load_dataset(
  'ILSVRC/imagenet-1k',
  split='train',
  streaming=True,
  trust_remote_code=True,
)

parser = ArgumentParser()
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-n', '--count', type=int, default=1000)
args = parser.parse_args()

for i, sample in enumerate(dataset.take(args.count)):
  image = sample['image']
  label = sample['label']

  class_path = os.path.join(args.output, str(label))
  os.makedirs(class_path, exist_ok=True)

  image_path = os.path.join(class_path, f'image_{i}.jpg')
  image.save(image_path)

print(f'Saved {args.count} images to {args.output}')
