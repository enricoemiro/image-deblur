from __future__ import annotations

import csv
import cv2
import numpy as np

class Analyzer():
  def __init__(self, data: dict) -> None:
    self.data = data

    self.metrics = {}

  def __calculate_average_std_psnr_mse(self) -> Analyzer:
    psnr_total = mse_total = 0
    psnr_arr = mse_arr = np.array([], dtype=np.double)

    for values in self.data.values():
      if 'metrics' in values:
        psnr, mse = values['metrics']

        psnr_total += psnr
        mse_total += mse

        psnr_arr = np.append(psnr_arr, psnr)
        mse_arr = np.append(mse_arr, mse)

    self.metrics |= {
      'psnr': {
        'average': psnr_total / len(psnr_arr),
        'std': np.std(psnr_arr)
      },
      'mse': {
        'average': mse_total / len(mse_arr),
        'std': np.std(mse_arr)
      }
    }

    return self

  def __save_images(self, path: str = '.') -> Analyzer:
    for key, values in self.data.items():
      image = values['image']
      cv2.imwrite(f'{path}/{key}.png', cv2.convertScaleAbs(image, alpha=(255.0)))

    return self

  def run(self, path: str = '.', filename: str = 'default') -> None:
    self.__calculate_average_std_psnr_mse() \
        .__save_images(path)

    with open(f'{path}/{filename}.csv', 'w') as file:
      writer = csv.writer(file)

      writer.writerow(['phase', 'psnr', 'mse'])
      for key, values in self.data.items():
        if 'metrics' in values:
          psnr, mse = values['metrics']
          writer.writerow([key, psnr, mse])

      writer.writerow([None,
                       f'avg = {self.metrics["psnr"]["average"]}',
                       f'avg = {self.metrics["mse"]["average"]}'])

      writer.writerow([None,
                       f'std = {self.metrics["psnr"]["std"]}',
                       f'std = {self.metrics["mse"]["std"]}'])

      file.close()
