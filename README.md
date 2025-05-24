# GAN MNIST Generator

This repository contains a simple PyTorch implementation of a GAN (Generative Adversarial Network) trained on the MNIST dataset to generate handwritten digit images.

## Features

- Uses a simple fully connected Generator and Discriminator.
- Trains with the MNIST dataset.
- Utilizes `accelerate` for easy multi-GPU/CPU training.
- Saves a final plot showing generated images for each epoch with labels.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- accelerate
- matplotlib

## Usage

1. Clone the repository.
2. Install dependencies:
```

pip install torch torchvision accelerate matplotlib

```
3. Run the training script:
```

python main.py

```
4. The generated images per epoch will be saved as `gan_generated_epochs_with_labels.png`.

## Notes

- Training uses a batch size of 128 for 20 epochs.
- Adjust hyperparameters in the script as needed.
- Ensure CUDA is available for faster training (optional).


