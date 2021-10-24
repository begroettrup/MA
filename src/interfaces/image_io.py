import torchvision.utils
import torchvision.transforms

from PIL import Image 

def save_grid(images, path, nrow=8):
  """
  Saves a list of images as an image.
  """
  grid = torchvision.utils.make_grid(images, nrow)
  torchvision.transforms.ToPILImage()(grid).save(path)
