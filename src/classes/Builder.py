import os
import sys
import pytz
import datetime

from skimage import io, data, draw

class Builder:
  def __init__(self, path: str, number_of_images: int) -> None:
    if not os.path.isdir(path):
      sys.exit(f'The path "{path}" does not exist.\u0020' \
                'Please create it before running the script.')

    self.path = path
    self.number_of_images = number_of_images

  def build(self) -> None:
    for i in range(self.number_of_images):
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
      io.imsave(f'{self.path}/{time}.png', image)

    # Extract from skimage.data:
    # - the image of the moon
    # - the image of human mitosis
    moon = data.moon()
    human_mitosis = data.human_mitosis()

    io.imsave(f'{self.path}/moon.png', moon)
    io.imsave(f'{self.path}/human_mitosis.png', human_mitosis)
