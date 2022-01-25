import os

from classes.PSF import PSF
from classes.Builder import Builder
from classes.Executor import Executor
from classes.Analyzer import Analyzer

def main():
  path = f'{os.getcwd()}/images'

  # Builder(path, 8).build()

  executor = Executor(path=f'{path}/6_1.png',
                      psf=PSF(0.5, 5),
                      llambda=0.1,
                      std_dev=0.01,
                      max_iterations=10)
  executor.run()

  analyzer = Analyzer(executor.data)
  analyzer.run()

if __name__ == '__main__':
  main()
