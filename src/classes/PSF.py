import numpy as np

from utilities.base import psf_fft, gaussian_kernel

class PSF:
  def __init__(self, sigma: np.double, kernlen: np.uint32) -> None:
    self.sigma = sigma
    self.kernlen = kernlen

  def K(self, shape: tuple) -> np.ndarray:
    return psf_fft(gaussian_kernel(self.kernlen, self.sigma), self.kernlen, shape)
