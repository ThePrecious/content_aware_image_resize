import logging
logging.getLogger("torch").setLevel(logging.ERROR)

import os, sys; 
sys.path.append("../../")

import numpy as np
import torch

from PIL import Image
from unet import *
from ml_model import *


# load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('./best_model.pth', map_location='cpu')
model = model.to(device, dtype=torch.float)
model.eval()

h = 384
w = 496
np_image = np.zeros((h,w), dtype='uint8')

#download test image
url = 'http://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png'
resp = urlopen(url)
np_image = np.asarray(bytearray(resp.read()), dtype="uint8")
resized_image = run_inference(np_image, model)

# assert image size 
assert (resized_image.size == h * 400 * 3)

# assert resized shaped image
image = Image.fromarray(resized_image.astype('uint8'))
assert (image.size == (400, 384))

print ('test case pass')
