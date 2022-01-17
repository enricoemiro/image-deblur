import numpy as np

from utils import total_variation_f, total_variation_df, o_minimize, generate_metrics

def fourth(image, K, llambda):
  x0 = np.zeros(image.shape)
  result = o_minimize(x0, image, K, llambda, total_variation_f, total_variation_df)
  total_variation_image = result[0]
  metrics = generate_metrics(image, total_variation_image)

  return total_variation_image, metrics
