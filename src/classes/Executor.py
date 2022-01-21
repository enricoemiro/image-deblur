from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import classes.PSF as PSF

from utilities.base import *
from utilities.general import *

from scipy.optimize import minimize

class Executor:
  def __init__(self, path: str, psf: PSF, llambda: np.float64, std_dev: np.float64, max_iterations: np.uint32 = 50) -> None:
    self.data = {}
    self.llambda = llambda
    self.std_dev = std_dev
    self.max_iterations = max_iterations

    self.__initial_setup(path, psf)

  def __initial_setup(self, path: str, psf: PSF) -> None:
    original_image = normalize_zero_one(plt.imread(path).astype(np.float64))
    self.K = psf.K(original_image.shape)

    self.data |= {
      'original': {
        'title': 'Immagine originale',
        'image': original_image,
      }
    }

  def __calculate_psnr_mse(self, image_after: np.ndarray) -> tuple:
    return calculate_psnr_mse(self.data['original']['image'], image_after)

  def first(self, image: np.ndarray) -> Executor:
    corrupted_image = add_blur_and_noise(image, self.K, self.std_dev)

    self.data |= {
      'corrupted': {
        'title': 'Immagine corrotta',
        'image': corrupted_image,
        'metrics': self.__calculate_psnr_mse(corrupted_image)
      }
    }

    return self

  def second(self, image: np.ndarray) -> Executor:
    m, n = image.shape

    f = lambda x : base_f(x.reshape(m, n), self.K, image)
    df = lambda x : base_df(x.reshape(m, n), self.K, image).reshape(m * n)

    x0 = np.zeros_like(image)
    result = minimize(fun=f, x0=x0, method='CG', jac=df, options={'maxiter': self.max_iterations})
    naif_image = np.reshape(result.x, (m, n))

    self.data |= {
      'naïf': {
        'title': 'Immagine naïf',
        'image': naif_image,
        'metrics': self.__calculate_psnr_mse(naif_image)
      }
    }

    return self

  def third(self, image: np.ndarray, our_minimize = True) -> Executor:
    m, n = image.shape
    x0 = np.zeros_like(image)

    if our_minimize:
      result = o_minimize(x0, image, self.K, self.llambda, regularization_f, regularization_df, max_iterations=self.max_iterations)
      regularized_image = result[0]

      self.data |= {
        'our_regularization': {
          'title': 'Immagine regolarizzata (our)',
          'image': regularized_image,
          'metrics': self.__calculate_psnr_mse(regularized_image),
        }
      }
    else:
      f = lambda x : regularization_f(x.reshape(m, n), self.K, image, self.llambda)
      df = lambda x : regularization_df(x.reshape(m, n), self.K, image, self.llambda).reshape(m * n)

      result = minimize(fun=f, x0=x0, method='CG', jac=df, options={'maxiter': self.max_iterations})
      regularized_image = np.reshape(result.x, (m, n))

      self.data |= {
        'lib_regularization': {
          'title': 'Immagine regolarizzata (lib)',
          'image': regularized_image,
          'metrics': self.__calculate_psnr_mse(regularized_image),
        }
      }

    return self

  def fourth(self, image: np.ndarray) -> Executor:
    x0 = np.zeros_like(image)
    result = o_minimize(x0, image, self.K, self.llambda, total_variation_f, total_variation_df, max_iterations=self.max_iterations)
    total_variation_image = result[0]

    self.data |= {
      'total_variation': {
        'title': 'Immagine variazione totale',
        'image': total_variation_image,
        'metrics': self.__calculate_psnr_mse(total_variation_image)
      }
    }

    return self

  def plot(self) -> None:
    figure = plt.figure(figsize=(8, 8))

    position = 1
    for v in self.data.values():
      figure.add_subplot(2, 3, position)
      plt.axis('off')

      title = v['title']
      if 'metrics' in v:
        psnr, mse = v['metrics']
        title += f'\nPSNR: {psnr:.10f}\nMSE: {mse:.10f}'

      plt.title(title)
      plt.imshow(v['image'], cmap='gray', vmin=0, vmax=1)

      position += 1

    plt.show()

  def run(self) -> None:
    original_image = self.data['original']['image']
    self.first(original_image)

    corrupted_image = self.data['corrupted']['image']
    self.second(corrupted_image)       \
        .third(corrupted_image)        \
        .third(corrupted_image, False) \
        .fourth(corrupted_image)       \
        .plot()
