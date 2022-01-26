import os
import numpy as np

from classes.PSF import PSF
from classes.Executor import Executor
from classes.Analyzer import Analyzer

def auto_run(path: str,
             psf: PSF,
             std_dev: np.float64,
             lambdas: list[np.float64],
             iterations: list[int],
             images_path: list[str]):
  base_dir = f'{path}/{psf.sigma}-{psf.kernlen}-{std_dev}'
  os.makedirs(base_dir, exist_ok=True)

  for llambda in lambdas:
    llambda_dir = f'{base_dir}/{llambda}'
    os.makedirs(llambda_dir, exist_ok=True)

    for iteration in iterations:
      iteration_dir = f'{llambda_dir}/{iteration}'
      os.makedirs(iteration_dir, exist_ok=True)

      for image in images_path:
        image_dir = f'{iteration_dir}/{os.path.splitext(os.path.basename(image))[0]}'
        os.makedirs(image_dir, exist_ok=True)

        executor = Executor(path=image,
                            psf=psf,
                            llambda=llambda,
                            std_dev=std_dev,
                            max_iterations=iteration)
        executor.run()

        analyzer = Analyzer(executor.data)
        analyzer.run(image_dir)
