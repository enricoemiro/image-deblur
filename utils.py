import numpy as np

from numpy import fft
from skimage import metrics

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

def generate_metrics(image_before: np.array, image_after: np.array) -> tuple:
  PSNR = metrics.peak_signal_noise_ratio(image_before, image_after)
  MSE = metrics.mean_squared_error(image_before, image_after)

  return (PSNR, MSE)

def next_step(x_k, K, b, llambda, grad, f):
  alpha = 1.0
  c1 = 0.25
  rho = 0.5
  max_iterations = 10

  p_k = -grad
  m, n = b.shape
  gradTp_k = np.dot(np.reshape(grad.T, m * n), np.reshape(p_k, m * n))

  armijo_expr_left = lambda x_k : f(x_k, K, b, llambda)
  armijo_expr_right = lambda x_k, alpha, c1 : \
                        f(x_k, K, b, llambda) + c1 * alpha * gradTp_k
  armijo = armijo_expr_left(x_k + alpha * p_k) > armijo_expr_right(x_k, alpha, c1)

  j = 0
  while (armijo and j < max_iterations):
    alpha *= rho
    j += 1

  return alpha

# Our version of the gradient method for minimizing the function f:
#   - x0 = initial guess of the algorithm
#   - b = the matrix of the corrupted image
#   - step = the length of the step in the direction p_k
#   - K = operator applying the Gaussian blur
#   - llambda = regularization parameter to reduce the effects of noise
def o_minimize(x0, b, K, llambda, f, df, absolute_stop = 1.e-5, max_iterations = 10):
  # x_k is the matrix at the k-th iteration
  x_k = np.copy(x0)

  norm_grad_list = \
    function_eval_list = \
      error_list = np.zeros((1, max_iterations))

  # Function for variation of the algorithm parameters during execution:
  #   - grad =
  #   - norm_grad = values of the gradient norm
  #   - function_eval = values of the objective function
  #   - error = solution error
  grad = lambda x_k : df(x_k, K, b, llambda)
  norm_grad = lambda x_k : np.linalg.norm(grad(x_k), 2)
  function_eval = lambda x_k : f(x_k, K, b, llambda)
  error = lambda x_k : np.linalg.norm(x_k - b, 2)

  # Function that updates the lists that contain
  # useful parameters for observing the algorithm's progress
  def update_lists(k: int):
    norm_grad_list[:, k] = norm_grad(x_k)
    function_eval_list[:, k] = function_eval(x_k)
    error_list[:, k] = error(x_k)

  k = 0
  while norm_grad(x_k) > absolute_stop and k < max_iterations:
    update_lists(k)

    g = grad(x_k)

    step = next_step(x_k, K, b, llambda, g, f)

    if step == -1:
      print('Not convergent')
      return k

    x_k = x_k - step * g

    k += 1

  return (x_k,
          norm_grad_list,
          function_eval_list,
          error_list,
          k)

def base_f(x, K, b):
  return (1 / 2) * (np.linalg.norm(A(x, K) - b, 2) ** 2)

def base_df(x, K, b):
  return AT(A(x, K), K) - AT(b, K)

def regularization_f(x, K, b, llambda):
  return base_f(x, K, b) + ((llambda / 2) * (np.linalg.norm(x, 2) ** 2))

def regularization_df(x, K, b, llambda):
  return base_df(x, K, b) + (llambda * x)
