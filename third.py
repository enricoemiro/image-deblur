import numpy as np

from utils import *
from scipy.optimize import minimize

# Function for the third part of the project:
#
# To reduce the effects of noise in the reconstruction it is
# necessary to introduce a Tikhonov regularization term.
#
# The function f to be minimized becomes:
#   1/2 * np.linalg.norm(A(x, K) - b, 2) ** 2 + lamb/2 * np.linalg.norm(x, 2)
#
# And the gradient function becomes:
#   ∇f(x) = ATAx - ATb + λx
def third(image, K, llambda, our_minimize = True):
  m, n = image.shape

  x0 = np.zeros(image.shape)
  result = regularized_image = None

  if our_minimize:
    result = o_minimize(x0, image, K, llambda, regularization_f, regularization_df)
    regularized_image = result[0]
  else:
    f = lambda x : regularization_f(x.reshape(m, n), K, image, llambda)
    df = lambda x : regularization_df(x.reshape(m, n), K, image, llambda).reshape(m * n)

    result = minimize(fun=f, x0=x0, method='CG', jac=df, options={'maxiter': 20})
    regularized_image = np.reshape(result.x, (m, n))

  metrics = generate_metrics(image, regularized_image)

  return regularized_image, metrics
