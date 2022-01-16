import numpy as np

from scipy.optimize import minimize
from utils import base_f, base_df, generate_metrics

# 2) Naive solution
#
# A possible reconstruction of the original image x starting from
# the corrupted image b is the naive solution given by the minimum
# of the following optimization problem:
#     1/2 * np.linalg.norm(A(x, K) - b, 2) ** 2
#
# And using the following gradient function:
#         ∇f(x) = ATAx − ATb
def second(image, K):
  m, n = image.shape

  f = lambda x : base_f(x.reshape(m, n), K, image)
  df = lambda x : base_df(x.reshape(m, n), K, image).reshape(m * n)

  x0 = np.zeros(image.shape)
  result = minimize(fun=f, x0=x0, method='CG', jac=df, options={'maxiter': 20})
  naive_image = np.reshape(result.x, (m, n))
  metrics = generate_metrics(image, naive_image)

  return naive_image, metrics
