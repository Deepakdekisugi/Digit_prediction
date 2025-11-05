"""Debug script to test image preprocessing and compare with MNIST"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image

# Load a sample MNIST image
(x_train, y_train), _ = mnist.load_data()

# Show original MNIST digit
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('MNIST Samples (what the model expects)')

for i in range(5):
    axes[0, i].imshow(x_train[i], cmap='gray')
    axes[0, i].set_title(f'Label: {y_train[i]}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(x_train[i+5], cmap='gray')
    axes[1, i].set_title(f'Label: {y_train[i+5]}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png')
print("Saved MNIST samples to mnist_samples.png")
print(f"MNIST image shape: {x_train[0].shape}")
print(f"MNIST pixel range: {x_train[0].min()} to {x_train[0].max()}")
print(f"MNIST mean: {x_train[0].mean():.2f}")
