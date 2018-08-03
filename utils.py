from PIL import Image, ImageDraw, ImageFont
import skimage.io as io
import numpy as np
import string
import random
import binascii


def generator_img_secret(nb_image=3300, img_size=128, txt_size=1):
	'''
	Generates secret images to hide in other images
	'''
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
	'''
	Generates secret text represented in a binary numpy array
	Each character is coded with 7 bits
	'''
	for i in range(nb_text_vector):
		secret = np.random.choice(list(string.ascii_letters), txt_size, replace=True)
		secret_string = ''.join(secret)
		binary_string = ''.join(format(ord(x), 'b') for x in secret_string)
		binary_array = list(binary_string)
		np.save('./generated_secret/binary_txt_secret/generated_txt_binary%d.npy'%i, np.asarray(binary_array))


def decrypt_binary_txt(path_to_img):
	'''
	Decode a binary numpy array as a plain text
	'''
	binary_array = np.load(path_to_img)
	binary_character_array = [binary_array[i * 7:(i + 1) * 7] for i in range((len(binary_array) + 7 - 1) // 7 )] 
	charac_stack = []
	
	for character in binary_character_array:
		n = int(''.join(character), 2)
		charac_stack.append(binascii.unhexlify('%x' % n).decode('ascii'))
	
	decoded_txt = ''.join(charac_stack)
	return decoded_txt


# generate_binary_from_txt()
# decrypt_binary_txt('./generated_secret/binary_txt_secret/generated_txt_binary0.npy')