import numpy as np

from numpy import fft

# Create a Gaussian kernel of size kernlen and standard deviation sigma
def gaussian_kernel(kernlen, sigma):
  x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)
  # Unidimensional Gaussian kernel
  kern1d = np.exp(- 0.5 * (x**2 / sigma))
  # Bidimensional Gaussian kernel
  kern2d = np.outer(kern1d, kern1d)
  # Normalization
  return kern2d / kern2d.sum()

# Compute the FFT of the kernel 'K' of size 'd' padding with the zeros necessary
# to match the size of 'shape'
def psf_fft(K, d, shape):
  # Zero padding
  K_p = np.zeros(shape)
  K_p[:d, :d] = K

  # Shift
  p = d // 2
  K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

  # Compute FFT
  K_otf = fft.fft2(K_pr)
  return K_otf

# Multiplication by A
def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))

# Multiplication by A transpose
def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))

# Total variation
def total_variation(x, eps = 1e-2):
  # Calculate the gradient of x
  dx, dy = np.gradient(x)
  n2 = np.square(dx) + np.square(dy)

  # Calculate the total variation of x
  tv = np.sqrt(n2 + eps**2).sum()
  return tv

# Gradient of the total variation
def grad_total_variation(x, eps = 1e-2):
  # Calculate the numerator of the fraction
  dx, dy = np.gradient(x)

  # Calculate the denominator of the fraction
  n2 = np.square(dx) + np.square(dy)
  den = np.sqrt(n2 + eps**2)

  # Calculate the two components of F by dividing the gradient by the denominator
  Fx = dx / den
  Fy = dy / den

  # Calculate the horizontal derivative of Fx
  dFdx = np.gradient(Fx, axis=0)

  # Calculate the vertical derivative of Fy
  dFdy = np.gradient(Fy, axis=1)

  # Calculate the divergence
  div = (dFdx + dFdy)

  # Return the value of the gradient of the total variation
  return -div

def base_f(x, K, b):
  return (1 / 2) * (np.linalg.norm(A(x, K) - b, 2) ** 2)

def base_df(x, K, b):
  return AT(A(x, K), K) - AT(b, K)

def regularization_f(x, K, b, llambda):
  return base_f(x, K, b) + ((llambda / 2) * (np.linalg.norm(x, 2) ** 2))

def regularization_df(x, K, b, llambda):
  return base_df(x, K, b) + (llambda * x)

def total_variation_f(x, K, b, llambda):
  return base_f(x, K ,b) + (llambda * total_variation(b))

def total_variation_df(x, K, b, llambda):
  return base_df(x, K, b) + (llambda * grad_total_variation(x))
