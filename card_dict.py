def card():
	# card1 = r,g,b
	card1 = {'first_line': [[178, 125, 66], [183, 151, 82], [151, 152, 110], [144, 151, 118], [142, 152, 127], [115, 135, 126], [108, 129, 124]], 'second_line': [[190, 180, 51], [166, 159, 50], [150, 148, 72], [126, 141, 84], [113, 135, 89], [103, 127, 96], [73, 113, 112]], 'third_line': [[149, 151, 67], [135, 143, 69], [111, 118, 85], [110, 116, 97], [92, 97, 93]], 'fourth_line': [[159, 141, 59], [150, 122, 51], [135, 94, 44], [116, 66, 45], [100, 55, 45], [99, 47, 46]], 'reference': [185, 185, 187]}
	return card1

def grayscale_one_pixel(RGB):
	gray = 0.2126 * RGB[0] + 0.7152 * RGB[1] + 0.0722 * RGB[2]
	return gray

def grayscal_card_array():
	RGB_card = card()
	fir_line_grays = []
	sec_line_grays = []
	thr_line_grays = []
	four_line_grays = []
	for i in RGB_card['first_line']:
		fir_line_grays.append(grayscale_one_pixel(i))
	for i in RGB_card['second_line']:
		sec_line_grays.append(grayscale_one_pixel(i))
	for i in RGB_card['third_line']:
		thr_line_grays.append(grayscale_one_pixel(i))
	for i in RGB_card['fourth_line']:
		four_line_grays.append(grayscale_one_pixel(i))
	# print(fir_line_grays)
	# print(sec_line_grays)
	# print(thr_line_grays)
	# print(four_line_grays)
	grayscal_card_array = [fir_line_grays, sec_line_grays, thr_line_grays, fir_line_grays, four_line_grays]
	return grayscal_card_array

def array_by_line(card):
	R_of_first_line = []
	G_of_first_line = []
	B_of_first_line = []
	R_of_second_line = []
	G_of_second_line = []
	B_of_second_line = []
	R_of_third_line = []
	G_of_third_line = []
	B_of_third_line = []
	R_of_fourth_line = []
	G_of_fourth_line = []
	B_of_fourth_line = []

	for i in range(0, 6):
		R_of_first_line.append(card['first_line'][i][0])
		G_of_first_line.append(card['first_line'][i][1])
		B_of_first_line.append(card['first_line'][i][2])
	first_color_by_RGB = [R_of_first_line, G_of_first_line, B_of_first_line]

	for i in range(0, 6):
		R_of_second_line.append(card['second_line'][i][0])
		G_of_second_line.append(card['second_line'][i][1])
		B_of_second_line.append(card['second_line'][i][2])
	second_color_by_RGB = [R_of_first_line, G_of_first_line, B_of_first_line]

	for i in range(0, 4):
		R_of_third_line.append(card['third_line'][i][0])
		G_of_third_line.append(card['third_line'][i][1])
		B_of_third_line.append(card['third_line'][i][2])
	third_color_by_RGB = [R_of_first_line, G_of_first_line, B_of_first_line]

	for i in range(0, 5):
		R_of_fourth_line.append(card['fourth_line'][i][0])
		G_of_fourth_line.append(card['fourth_line'][i][1])
		B_of_fourth_line.append(card['fourth_line'][i][2])
	fourth_color_by_RGB = [R_of_first_line, G_of_first_line, B_of_first_line]

	return first_color_by_RGB, second_color_by_RGB, third_color_by_RGB, fourth_color_by_RGB

if __name__ == '__main__':
	linspace()

