import numpy as np

from utilities.base import A
from skimage import metrics

def normalize_zero_one(image: np.ndarray) -> np.ndarray:
  return (image - np.amin(image)) / (np.amax(image) - np.amin(image))

def add_blur_and_noise(image: np.ndarray, K: np.ndarray, std_dev: np.double) -> np.ndarray:
  noise = np.random.normal(size=image.shape) * std_dev

  return A(image, K) + noise

def calculate_psnr_mse(image_before: np.ndarray, image_after: np.ndarray) -> tuple:
  PSNR = metrics.peak_signal_noise_ratio(image_before, image_after, data_range=1)
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

  armijo_left = lambda x_k : f(x_k, K, b, llambda)
  armijo_right = lambda x_k, alpha, c1 : f(x_k, K, b, llambda) + c1 * alpha * gradTp_k
  armijo = armijo_left(x_k + alpha * p_k) > armijo_right(x_k, alpha, c1)

  j = 0
  while (armijo and j < max_iterations):
    alpha *= rho
    j += 1

  return alpha

def o_minimize(x0, b, K, llambda, f, df, absolute_stop = 1.e-5, max_iterations = 10):
  grad = lambda x_k : df(x_k, K, b, llambda)
  norm_grad = lambda x_k : np.linalg.norm(grad(x_k), 2)
  function_eval = lambda x_k : f(x_k, K, b, llambda)
  error = lambda x_k : np.linalg.norm(x_k - b, 2)

  x_k = np.copy(x0)

  norm_grad_list = \
    function_eval_list = \
      error_list = np.zeros((1, max_iterations))

  k = 0
  while norm_grad(x_k) > absolute_stop and k < max_iterations:
    norm_grad_list[:, k] = norm_grad(x_k)
    function_eval_list[:, k] = function_eval(x_k)
    error_list[:, k] = error(x_k)

    g = grad(x_k)

    step = next_step(x_k, K, b, llambda, g, f)

    if step == -1:
      print('The method does not converge')
      return k

    x_k = x_k - step * g

    k += 1

  return (x_k, norm_grad_list, function_eval_list, error_list, k)
