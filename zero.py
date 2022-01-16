import os
import sys
import pytz
import datetime

from skimage import io, data, draw
from constant import DATASET_SIZE, BASE_DIR

def generate_dataset(number_of_images: int, path: str) -> None:
  for i in range(number_of_images):
    image = draw.random_shapes((512, 512),
                                max_shapes=6,
                                min_shapes=2,
                                min_size=100,
                                max_size=200,
                                multichannel=False,
                                allow_overlap=True)[0]

    # Set the background from white to black
    image[image == 255] = 0

    # Export the image in .png format
    time = datetime.datetime.now(tz=pytz.timezone('Europe/Rome'))
    io.imsave(f'{path}/{time}.png', image)

  # Extract from skimage.data:
  # - the image of the moon
  # - the image of human mitosis
  moon = data.moon()
  human_mitosis = data.human_mitosis()

  io.imsave(f'{path}/moon.png', moon)
  io.imsave(f'{path}/human_mitosis.png', human_mitosis)

def zero(number_of_images: int, path: str, generate: bool = False):
  if not os.path.isdir(path):
    sys.exit(f'The path "{path}" does not exist.\u0020' \
              'Please create it before running the script.')

  if generate == True:
    generate_dataset(number_of_images, path)

# Uncomment the following line to run the code.
#
# zero(DATASET_SIZE, BASE_DIR, True)
