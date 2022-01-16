import numpy as np

from utils import *

# Add blur and noise to the given image
def add_blur_and_noise(image: np.array, K, standard_deviation):
  noise = np.random.normal(size=image.shape) * standard_deviation

  return A(image, K) + noise

# 1) Corrupted image generation
#
# Degrade the images by applying the blur operator with parameters:
#   - σ = 0.5   dimension  5×5
#   - σ = 1     dimension  7×7
#   - σ = 1.3   dimension  9×9
# and adding Gaussian noise with standard deviation (0.0, 0.05]
def first(image: np.array, K, standard_deviation):
  corrupted_image = add_blur_and_noise(image, K, standard_deviation)
  metrics = generate_metrics(image, corrupted_image)

  return corrupted_image, metrics
