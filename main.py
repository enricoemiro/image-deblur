import os
import numpy as np
import matplotlib.pyplot as plt

from zero import zero
from first import first
from second import second
from third import third
from fourth import fourth
from utils import psf_fft, gaussian_kernel

figure = plt.figure(figsize=(20, 20), frameon=False)

def normalize_zero_one(image) -> np.array:
  return (image - np.amin(image)) / (np.amax(image) - np.amin(image))

def write_metrics(file, metrics):
  file.write(f'{metrics[0]},{metrics[1]}\n')

def add_to_figure(image: np.array, title: str, position: int, metrics = None):
  figure.add_subplot(2, 3, position)
  plt.axis('off')

  if metrics == None: plt.title(title)
  else: plt.title(f'{title}\nPSNR: {metrics[0]:.10f}\nMSE: {metrics[1]:.10f}')

  plt.imshow(image, cmap='gray', vmin=0, vmax=1)

def run(path: str, sigma, kernlen, standard_deviation, llambda, show_plot = True, save_data = True):
  # Load the image from the path and calculate K
  original_image = normalize_zero_one(plt.imread(path, format='png').astype(np.float64))
  K = psf_fft(gaussian_kernel(kernlen, sigma), kernlen, original_image.shape)

  # Phase 1 to 4
  corrupted_image, corrupted_metrics = first(original_image, K, standard_deviation)
  naive_image, naive_metrics = second(corrupted_image, K)
  regularized_image_0, regularized_metrics_0 = third(corrupted_image, K, llambda)
  regularized_image_1, regularized_metrics_1 = third(corrupted_image, K, llambda, False)
  total_variation_image, total_variation_metrics = fourth(corrupted_image, K, llambda)

  if save_data:
    splitted_path, filename = os.path.split(path)
    filename_without_extension = os.path.splitext(filename)[0]
    data = open(f'{splitted_path}/metrics/{filename_without_extension}-metrics.csv', 'w')

    data.write('psnr,mse\n')
    write_metrics(data, corrupted_metrics)
    write_metrics(data, naive_metrics)
    write_metrics(data, regularized_metrics_0)
    write_metrics(data, regularized_metrics_1)
    write_metrics(data, total_variation_metrics)

  if show_plot:
    # Add all the images to the figure
    add_to_figure(original_image, 'Immagine originale', 1)
    add_to_figure(corrupted_image, 'Immagine corrotta', 2, corrupted_metrics)
    add_to_figure(naive_image, 'Immagine naive', 3, naive_metrics)
    add_to_figure(regularized_image_0, 'Immagine regolarizzata (o_minimize)', 4, regularized_metrics_0)
    add_to_figure(regularized_image_1, 'Immagine regolarizzata (minimize)', 5, regularized_metrics_1)
    add_to_figure(total_variation_image, 'Immagine variazione totale', 6, total_variation_metrics)

    # Plot all images
    plt.show()

def main():
  # Uncomment the following line to generate the dataset
  #
  # zero(8, 'images', True)

  run('images/6_1.png', 0.5, 5, 0.05, 0.01)

if __name__ == '__main__':
  main()
