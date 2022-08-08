# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import math
import numpy as np
from card_dict import card, array_by_line, grayscale_one_pixel, grayscal_card_array
from scipy import stats
import imageio
from resizeimage import resizeimage
from PIL import Image

font = cv2.FONT_HERSHEY_COMPLEX

def image_preprocessing(img, width, height, gray_para, blur_para, show_name, d):
	# resized_img = imutils.resize(img, width)
	resized_img = cv2.resize(img, (width, height))
	ratio_img = resized_img.shape[0] / float(resized_img.shape[1])

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	blurred_img = cv2.GaussianBlur(gray_img, (gray_para[0], gray_para[1]), gray_para[2])
	thresh_img = cv2.threshold(blurred_img, blur_para[0], blur_para[1], cv2.THRESH_BINARY)[1]
	if d == 'debug':
		cv2.imshow("blurred_" + show_name, blurred_img)
		cv2.imshow("thresh_i" + show_name, thresh_img)
	return resized_img, ratio_img, gray_img, blurred_img, thresh_img

def thresh_img_preprocessing(img, width, height, gray_para, blur_para, show_name, d):
	# resized_img = imutils.resize(img, width)
	resized_img = cv2.resize(img, (width, height))
	ratio_img = resized_img.shape[0] / float(resized_img.shape[1])

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	blurred_img = cv2.GaussianBlur(gray_img, (gray_para[0], gray_para[1]), gray_para[2])
	thresh_img1 = cv2.threshold(blurred_img, blur_para[0], blur_para[1], cv2.THRESH_BINARY_INV)[1]
	kernel = np.ones((5,5),np.uint8)
	thresh_img = cv2.erode(thresh_img1,kernel,iterations = 3)
	if d == 'debug' or d == 'thresh':
		cv2.imshow("blurred_" + show_name, blurred_img)
		cv2.imshow("thresh_" + show_name, thresh_img)
		cv2.waitKey(0)
	return resized_img, ratio_img, gray_img, blurred_img, thresh_img

def find_max_contour(contours, ratio, resized_Cimg, d):
	sd = ShapeDetector()
	if len(contours) == 0:
		return
	else:
		# based on contour area, get the maximum contour which is the hand
		max_contour = max(contours, key=cv2.contourArea)
		
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(max_contour)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			shape = sd.detect(max_contour)

			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
			max_contour = max_contour.astype("float")
			# max_contour *= ratio
			max_contour = max_contour.astype("int")
			if d == "debug" or d == "max_contour":
				cv2.drawContours(resized_Cimg, [max_contour], -1, (0, 0, 0), 2)
				# cv2.putText(resized_Cimg, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.imshow("max_contour", resized_Cimg)
				cv2.waitKey(0)
		return max_contour

def findContours(img):

	# find contours in the thresholded image and initialize the
	# shape detector
	cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	return cnts

def is_valid__cnt(img, img_area, min_max_area_ratio, min_max_wh_ratio, contour, area, bounding_rect, d):
	if area == 0:
		return False
	else :
		area_ratio = img_area / area
		wh_ratio = bounding_rect[3] / bounding_rect[2]
		if min_max_area_ratio[0] <= area_ratio <= min_max_area_ratio[1] and min_max_wh_ratio[0] <= wh_ratio <= min_max_wh_ratio[1]:
			# img_box = cv2.rectangle(img, (bounding_rect[0], bounding_rect[1]), (bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]), color = (255, 0, 0), thickness = 1)
			# cv2.drawContours(img, [contour], -1, (0, 0, 0), 2)
			# cv2.imshow("is_valid__cnt", img)
			# cv2.waitKey(0)
			# print("+++++++++++++++", area_ratio, wh_ratio)
			return True
		
def find_valid_cnt(img, cnts, min_max_area_ratio, min_max_wh_ratio, d):
	height = img.shape[0]
	width = img.shape[1]
	img_area = width * height
	size_candidates = []
	candidates = []
	for contour in cnts:
		bounding_rect = cv2.boundingRect(contour)
		bounding_rect_area = bounding_rect[2] * bounding_rect[3]
		if is_valid__cnt(img, img_area, min_max_area_ratio, min_max_wh_ratio, contour, bounding_rect_area, bounding_rect, d):
			candidate = (bounding_rect[2] + bounding_rect[3]) / 2
			size_candidates.append(candidate)
			candidates.append(contour)
	return size_candidates, candidates

def valid_crop_img(img, min_max_ratio):
	h_crop_strip, w_crop_strip = img.shape[:2]
	ratio_crop_strip = h_crop_strip / w_crop_strip
	if min_max_ratio[0] <= ratio_crop_strip <= min_max_ratio[1]:
		return "yes"
	else: 
		"no"

def find_line(cnt, resized_Cimg, d):
	rows,cols = resized_Cimg.shape[:2]
	[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
	lefty = int((-x*vy/vx) + y)
	righty = int(((cols-x)*vy/vx)+y)
	if d == "debug":
		# cv2.line(resized_Cimg,(cols-1,righty),(0,lefty),(0,255,0),2)
		cv2.imshow("with line", resized_Cimg)
	point1 = [cols-1, righty]
	point2 = [0,lefty]
	return point1, point2

def caculate_angle(point1, point2):
	angle = math.atan2(abs(point2[1] - point1[1]), abs(point2[0] - point1[0]))
	angle = angle * 180 / 3.14
	return angle

def rotate_image(angle, resized_Cimg, d):
	image_center = tuple(np.array(resized_Cimg.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, 90 - angle, 1.0)
	result = cv2.warpAffine(resized_Cimg, rot_mat, resized_Cimg.shape[1::-1], flags=cv2.INTER_LINEAR)
	if d == "debug":
		cv2.imshow("rotate_image", result)
	return result

def crop_contour(contour, img, d):
	x,y,w,h = cv2.boundingRect(contour)
	cropped = img[int(y + h / 20):int(y + h - h / 20), int(x + w / 10):int(x + w - w / 10)]
	# cropped = img[int(y):int(y + h), int(x):int(x + w)]

	if d == "debug" or d == "crop_strip_contour":
		cv2.imshow("crop_contour", cropped)
		cv2.waitKey(0)
	return cropped

def takeSecond(elem):

	return elem[1]

def find_closest(x, array2d):
    least_diff = 999
    least_diff_index = -1
    for num, elm in enumerate(array2d):
        diff = abs(x[0]-elm[0]) + abs(x[1]-elm[1]) + abs(x[2]-elm[2])
        if diff < least_diff:
            least_diff = diff
            least_diff_index = num
    return array2d[least_diff_index], least_diff_index

def fine_nearest_array(array_2d, array):
	dist = lambda x, y: (x[0]-y[0])**2 + (x[1]-y[1])**2
	min(array_2d, key=lambda co: dist(co, array))

def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def fun_cnt_with_pos_array(cnt_candidates):
	pix_vals = []
	cnt_with_pos_array = []
	for cnt in cnt_candidates:

		#Getting topmost coordinates of contours.
		approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
			# Used to flatted the array containing 
			# the co-ordinates of the vertices. 
		n = approx.ravel() 
		i = 0
		for j in n : 
			if(i % 2 == 0): 
				x = n[i] 
				y = n[i + 1] 
				# String containing the co-ordinates. 
				string = str(x) + " " + str(y) 
				if(i == 0): 
					# print("x,y", x,y)
					cnt_with_pos = [cnt, y]
			i = i + 1
		## [[contour2, y2], [contour1, y1]....., [contour3, y3]]   here y is a size.
		cnt_with_pos_array.append(cnt_with_pos)
	## [[contour1, y1], [contour1, y1]....., [contour3, y1]]
	cnt_with_pos_array.sort(key=takeSecond)
	return cnt_with_pos_array

def get_mean_RGB_array(cnt_with_pos_img_array , cropped_img, d):
	color_rect_array = []
	mode_RGB_array = []
	for i in range(0, len(cnt_with_pos_img_array)):

		#Getting individual color rect
		cnt = cnt_with_pos_img_array[i][0]
		cropped_color_rect = crop_contour(cnt, cropped_img, d)
		color_rect_array.append(cropped_color_rect)

		#Getting the colors from color rect.
		rows, cols, RGB = cropped_color_rect.shape
		R_array = []
		G_array = []
		B_array = []
		for i in range(rows):
			for j in range(cols):
				RGB_pixel = cropped_color_rect[i,j]
				R_array.append(RGB_pixel[2])
				G_array.append(RGB_pixel[1])
				B_array.append(RGB_pixel[0])
		R_mode = stats.mode(R_array)[0][0]
		G_mode = stats.mode(G_array)[0][0]
		B_mode = stats.mode(B_array)[0][0]
		mode_RGB = [R_mode, G_mode, B_mode]
		mode_RGB_array.append(mode_RGB)
	return mode_RGB_array

def get_mean_RGB_reference(cropped_color_rect, d):

	#Getting the colors from color rect.
	rows, cols, RGB = cropped_color_rect.shape
	R_array = []
	G_array = []
	B_array = []
	for i in range(rows):
		for j in range(cols):
			RGB_pixel = cropped_color_rect[i,j]
			R_array.append(RGB_pixel[2])
			G_array.append(RGB_pixel[1])
			B_array.append(RGB_pixel[0])
			# print("@@@@@@@@@@@@@@", len(RGB_pixel))
	R_mode = stats.mode(R_array)[0][0]
	G_mode = stats.mode(G_array)[0][0]
	B_mode = stats.mode(B_array)[0][0]
	mode_RGB = [R_mode, G_mode, B_mode]
	return mode_RGB

def main(Cimg, d):

	##**************** Getting Mode values of 4 rects color  *******************
	# Processing Rimg and Cimg.
	# Getting rotate img.
	Cimg_gray_para = [3, 3, 0]
	Cimg_blur_para = [150, 255]
	resized_Cimg, ratio_Cimg, gray_Cimg, blurred_Cimg, thresh_Cimg = image_preprocessing(Cimg, 300, 500, Cimg_gray_para, Cimg_blur_para, 'Cimg', d)
	cnts_Cimg = findContours(thresh_Cimg)
	strip_min_max_area_ratio = [1.2, 12] 
	strip_min_max_wh_ratio = [2, 15] 

	size_candidate, cnt_Cimgs = find_valid_cnt(resized_Cimg, cnts_Cimg, strip_min_max_area_ratio, strip_min_max_wh_ratio, d)
	if len(cnt_Cimgs) == 0:
		h, w = Cimg.shape[:2]
		cropped_img = Cimg[0 : h, int(2*w/5) : int(3*w/5)]
		# cropped_img = Cimg[0 : h, int(w/2 - 20) : int(w/2 + 20)]

	else: 
		if len(cnt_Cimgs) == 1:
			cnt_Cimg = cnt_Cimgs[0]
			if d == "valid_teststrip" or d == "valid_teststrip":
				cv2.drawContours(resized_Cimg, [cnt_Cimgs[0]], -1, (0, 0, 0), 2)	
				cv2.imshow("is_valid_teststrip_cnt", resized_Cimg )
				cv2.waitKey(0)
			point1_Cimg, point2_Cimg = find_line(cnt_Cimg, resized_Cimg, d)
			angle_Cimg = caculate_angle(point1_Cimg, point2_Cimg)
			rotated_img = rotate_image(angle_Cimg, resized_Cimg, d)

			# Processing rotated_img.
			# Getting cropped img from rotated_img.
			rotated_img_gray_para = [3, 3, 0]
			rotated_img_blur_para = [130, 255]
			resized_rotated_img, ratio_rotated_img, gray_rotated_img, blurred_rotated_img, thresh_rotated_img = image_preprocessing(rotated_img, 300, 500, rotated_img_gray_para, rotated_img_blur_para, 'rotate_img', d)
			cnts_rotated = findContours(thresh_rotated_img)
			cnt_rotated_img = find_max_contour(cnts_rotated, ratio_rotated_img, resized_rotated_img, d)
			cropped_img = crop_contour(cnt_rotated_img, resized_rotated_img, d)
			# h_crop_strip, w_crop_strip = cropped_img.shape[:2]
			# ratio_crop_strip = h_crop_strip / w_crop_strip
			min_max_ratio = [6, 18]
			valid = valid_crop_img(cropped_img, min_max_ratio)
			if valid == "yes":
				cropped_img = cropped_img
			else: 
				print("There is a suitable contour but not any suitable cropped images.")
				return "Please retry on black background."
		else:
			print("There are too many contours.")		
			return "Please retry on black background."
	
	if d == "debug" or d == "crop_strip_img" or d == "debug1":
		cropped_img = cv2.resize(cropped_img, (50, 500))
		cv2.imshow("crop_strip_img", cropped_img)	
		cv2.waitKey(0)	
	# Processing the cropped_img for color rect.
	cropped_img = cv2.resize(cropped_img, (50, 500))
	h_cropped_img, w_cropped_img, c_cropped_img = cropped_img.shape
	cropped_img_gray_para_rect = [5, 5, 1]
	cropped_img_blur_para_rect = [180, 255]
	resized_cropped_img, ratio_cropped_img, gray_cropped_img, blurred_cropped_img, thresh_cropped_img = thresh_img_preprocessing(cropped_img, w_cropped_img, h_cropped_img, cropped_img_gray_para_rect, cropped_img_blur_para_rect, 'cropped_img', d)
	
	cnts_color_rect = findContours(thresh_cropped_img)
	# rect_min_max_area_ratio = [8, 20]
	# rect_min_max_wh_ratio = [0.8, 4]
	rect_min_max_area_ratio = [0, 50]
	rect_min_max_wh_ratio = [0.1, 20]
	size_candidates, cnt_candidates = find_valid_cnt(cropped_img, cnts_color_rect, rect_min_max_area_ratio, rect_min_max_wh_ratio, 'valid_rect_contour')
	if len(cnt_candidates) < 3:
		print("There aren't enough contours.")
		return "Please retry on black background."
	# Getting cropped color_rect and pixel values from cropped_img.
	cnt_with_pos_img_array = fun_cnt_with_pos_array(cnt_candidates)
	# Getting  color_rect with mode RGB
	mean_RGB_array = get_mean_RGB_array(cnt_with_pos_img_array, cropped_img, d)
	# print("=mean_RGB_array====", mean_RGB_array)
		
	## ****************  Getting Mode of reference color  *******************

	# Processing the cropped_img.
	h_cropped_img, w_cropped_img, c_cropped_img = cropped_img.shape
	cropped_img_gray_para = [5, 5, 1]
	cropped_img_blur_para = [185, 255]
	resized_cropped_img, ratio_cropped_img, gray_cropped_img, blurred_cropped_img, thresh_cropped_img = image_preprocessing(cropped_img, w_cropped_img, h_cropped_img, cropped_img_gray_para, cropped_img_blur_para, 'cropped_img_reference', d)
	cnts_color_rect = findContours(thresh_cropped_img)
	max_contour = find_max_contour(cnts_color_rect, ratio_cropped_img, resized_cropped_img, d)
	cropped_reference_img = crop_contour(max_contour, cropped_img, d)
	min_max_ratio = [2.5, 10] 
	valid_refer = valid_crop_img(cropped_reference_img, min_max_ratio)
	if valid_refer == "yes":
		mean_RGB_reference = get_mean_RGB_reference(cropped_reference_img, d)
	else:
		print("There isn't a suitable refernce rect.")
		return "Please retry on black background."

	# Getting the color of the test strip
	test_strip_mean_color = {"reference":mean_RGB_reference, "color_rect":mean_RGB_array}

	# Getting the colors of the card
	card_color = card()

	#************ Comparing between the reference of the card and one of the test strip.*************

	# Caculating ratio between the reference of the card and the one of the test strip

	ratio_R = card_color['reference'][0] / test_strip_mean_color['reference'][0]
	ratio_G = card_color['reference'][1] / test_strip_mean_color['reference'][1]
	ratio_B = card_color['reference'][2] / test_strip_mean_color['reference'][2]
	ratio_R = 1
	ratio_G = 1
	ratio_B = 1

	first_ratio_R_test_strip = test_strip_mean_color["color_rect"][0][0] * ratio_R
	first_ratio_G_test_strip = test_strip_mean_color["color_rect"][0][1] * ratio_G
	first_ratio_B_test_strip = test_strip_mean_color["color_rect"][0][2] * ratio_B
	first_ratio_RGB_test_strip = [first_ratio_R_test_strip, first_ratio_G_test_strip, first_ratio_B_test_strip]

	# print("+111++++", test_strip_mean_color["color_rect"])
	# print("+2222++++", test_strip_mean_color["color_rect"][1][0])
	second_ratio_R_test_strip = test_strip_mean_color["color_rect"][1][0] * ratio_R
	second_ratio_G_test_strip = test_strip_mean_color["color_rect"][1][1] * ratio_G
	second_ratio_B_test_strip = test_strip_mean_color["color_rect"][1][2] * ratio_B
	second_ratio_RGB_test_strip = [second_ratio_R_test_strip, second_ratio_G_test_strip, second_ratio_B_test_strip]

	third_ratio_R_test_strip = test_strip_mean_color["color_rect"][2][0] * ratio_R
	third_ratio_G_test_strip = test_strip_mean_color["color_rect"][2][1] * ratio_G
	third_ratio_B_test_strip = test_strip_mean_color["color_rect"][2][2] * ratio_B
	third_ratio_RGB_test_strip = [third_ratio_R_test_strip, third_ratio_G_test_strip, third_ratio_B_test_strip]

	fourth_ratio_R_test_strip = test_strip_mean_color["color_rect"][3][0] * ratio_R
	fourth_ratio_G_test_strip = test_strip_mean_color["color_rect"][3][1] * ratio_G
	fourth_ratio_B_test_strip = test_strip_mean_color["color_rect"][3][2] * ratio_B
	fourth_ratio_RGB_test_strip = [fourth_ratio_R_test_strip, fourth_ratio_G_test_strip, fourth_ratio_B_test_strip]

	first_near, first_index = find_closest(first_ratio_RGB_test_strip, card_color["first_line"])
	second_near, second_index = find_closest(second_ratio_RGB_test_strip, card_color["second_line"])
	third_near, third_index = find_closest(third_ratio_RGB_test_strip, card_color["third_line"])
	fourth_near, fourth_index = find_closest(fourth_ratio_RGB_test_strip, card_color["fourth_line"])

	# Finding the grayscal array from the Mode card.
	gray_card_array = grayscal_card_array()

	# Finding the grayscal array from the Mode test strip.
	first_ratio_gray_test_strip = grayscale_one_pixel(first_ratio_RGB_test_strip)
	second_ratio_gray_test_strip = grayscale_one_pixel(second_ratio_RGB_test_strip)
	third_ratio_gray_test_strip = grayscale_one_pixel(third_ratio_RGB_test_strip)
	fourth_ratio_gray_test_strip = grayscale_one_pixel(fourth_ratio_RGB_test_strip)

	# Finding the nearest gray value and index in grayscal Mode card.
	fir_nearest_value, fir_nearest_index = find_nearest_value(gray_card_array[0], first_ratio_gray_test_strip)
	sec_nearest_value, sec_nearest_index = find_nearest_value(gray_card_array[1], second_ratio_gray_test_strip)
	thir_nearest_value, thir_nearest_index = find_nearest_value(gray_card_array[2], third_ratio_gray_test_strip)
	four_nearest_value, four_nearest_index = find_nearest_value(gray_card_array[3], fourth_ratio_gray_test_strip)

	print(first_index + 1, second_index + 1, third_index + 1, fourth_index + 1)
	# print("first_ratio_RGB_test_strip", first_ratio_RGB_test_strip)
	# print('card_color["first_line"]', card_color["first_line"])
	# print("second_ratio_RGB_test_strip", second_ratio_RGB_test_strip)
	# print('card_color["second_line"]', card_color["second_line"])
	# print("third_ratio_RGB_test_strip", third_ratio_RGB_test_strip)
	# print('card_color["third_line"]', card_color["third_line"])
	# print("fourth_ratio_RGB_test_strip", fourth_ratio_RGB_test_strip)
	# print('card_color["fourth_line"]', card_color["fourth_line"])

	return test_strip_mean_color

if __name__ == '__main__':

	Cimg = cv2.imread("R_C_img/test1.jpg")

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--debug", required=False,
		help="path to the input image")
	args = vars(ap.parse_args())
	debug = args["debug"]
	main(Cimg, debug)


	
	