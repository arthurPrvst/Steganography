from PIL import Image, ImageDraw, ImageFont
import skimage.io as io
import numpy as np
import string
import random


def generator_img_secret(nb_image=3300, img_size=128, txt_size=1):

	for i in range(nb_image):
		img = Image.new('RGB', (img_size, img_size), color = (255, 255, 255))
		positionx = np.random.choice(np.arange(0,img_size-40,1), 1, replace=True)
		positiony = np.random.choice(np.arange(0,img_size-50,1), 1, replace=True)
		secret = np.random.choice(list(string.ascii_letters), txt_size, replace=True)
		draw = ImageDraw.Draw(img)
		font_path = './Roboto-Black.ttf'
		font = ImageFont.truetype(font_path, 40)
		draw.text((positionx, positiony), ''.join(secret), fill=(0,0,0), font=font)
		img.save('./generated_secret/img_secret/generated_image%d.png'%i)

def generate_binary_from_txt(nb_text_vector=1, txt_size=6):
	
	for i in range(nb_text_vector):
		