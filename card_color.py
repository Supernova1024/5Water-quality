# import the necessary packages
import cv2
import math
import numpy as np
import os
import sys
from scipy import stats
from card_dict import array_by_line
sys.path.append("..")

def get_mode_RGB_card(img):
	#Getting the colors from color rect.
	rows, cols, RGB = img.shape
	R_array = []
	G_array = []
	B_array = []
	for i in range(rows):
		for j in range(cols):
			RGB_pixel = img[i,j]
			R_array.append(RGB_pixel[2])
			G_array.append(RGB_pixel[1])
			B_array.append(RGB_pixel[0])
	R_mode = stats.mode(R_array)[0][0]
	G_mode = stats.mode(G_array)[0][0]
	B_mode = stats.mode(B_array)[0][0]
	mode_RGB = [R_mode, G_mode, B_mode]
	# print("==========", R_mode, mode_RGB)
	return mode_RGB, RGB

def card_color_main():
	path = 'training_data/2/'
	mode_RGB_card_array = []
	init_RGB_array = []
	mode_RGB_card_list = {}
	for img in os.listdir(path):
		if img.endswith(".png") or img.endswith(".jpg"):
			# if img != "r3.png":
			# print(img)
			image_name = os.path.join(path, img)
			image = cv2.imread(image_name)
			# Processing the cropped_img.
			mode_RGB_card, RGB = get_mode_RGB_card(image)
			init_RGB_array.append(RGB)
			mode_RGB_card_array.append(mode_RGB_card)
			mode_RGB_card_list[str(img)] = mode_RGB_card

	# print("==mean_RGB_array====", mode_RGB_card_array)
	# np.savetxt('mode_RGB_card_array.csv', mode_RGB_card_array, delimiter=',')
	# print("*********************************************")
	# print("==mean_RGB_list====", init_RGB_array)

	first_line = []
	second_line = []
	third_line = []
	fourth_line = []

	for i in range(0, len(mode_RGB_card_array)):
		if i <= 6:
			first_line.append(mode_RGB_card_array[i])
		elif 7 <= i <= 13:
			second_line.append(mode_RGB_card_array[i])
		elif 14 <= i <= 18:
			third_line.append(mode_RGB_card_array[i])
		elif 18 <= i <= 24:
			fourth_line.append(mode_RGB_card_array[i])
		else:
			reference = mode_RGB_card_array[i]
	card_dict = {}
	card_dict["first_line"] = first_line
	card_dict["second_line"] = second_line
	card_dict["third_line"] = third_line
	card_dict["fourth_line"] = fourth_line
	card_dict["reference"] = reference
	first_color_by_RGB, second_color_by_RGB, third_color_by_RGB, fourth_color_by_RGB = array_by_line(card_dict)

	print(card_dict)

	# print(first_color_by_RGB[0])
	return mode_RGB_card_array, init_RGB_array

if __name__ == '__main__':
	card_color_main()
	
	