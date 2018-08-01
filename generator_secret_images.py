from PIL import Image, ImageDraw, ImageFont
import skimage.io as io
import numpy as np
import string
import random

nb_image = 3300
img_size = 128


for i in range(nb_image):
	img = Image.new('RGB', (img_size, img_size), color = (255, 255, 255))
	positionx = np.random.choice(np.arange(0,img_size-40,1), 1, replace=True)
	positiony = np.random.choice(np.arange(0,img_size-20,1), 1, replace=True)
	secret = np.random.choice(list(string.ascii_letters), 6, replace=True)
	draw = ImageDraw.Draw(img)
	draw.text((positionx, positiony), ''.join(secret), fill=(0,0,0))
	img.save('./generated_secret/generated_image%d.png'%i)
