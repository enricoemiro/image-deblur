import os

from classes.PSF import PSF
from classes.Builder import Builder
from classes.Executor import Executor
from classes.Analyzer import Analyzer

def main():
  path = f'{os.getcwd()}/images'

  # Builder(path, 8).build()

  executor = Executor(f'{path}/6_1.png', PSF(0.5, 5), 0.1, 0.1, 10)
  executor.run()

  analyzer = Analyzer(executor.data)
  analyzer.run()

if __name__ == '__main__':
  main()
