try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def processImage(image):
    image = Image.open(image)
    image = ImageOps.grayscale(image)

    image = np.array(image)
    return image


def is_red_pixel(image, i, j):
    count_red = 0
    x_arr = [0, 1, 0, -1, 1, -1, -1, 1, 0]
    y_arr = [1, 0, -1, 0, 1, -1, 1, -1, 0]
    for k in range(9):
        if image[i - x_arr[k]][j - y_arr[k]][0] == 255:
            count_red += 1
    if count_red >= 6:
        return True
    return False


def find_tlf_lights_by_kernel(image, original_image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = sg.convolve(image, kernel, 'same')
    max_output = maximum_filter(output, size=25)

    x_red = list()
    y_red = list()
    x_green = list()
    y_green = list()

    for i in range(5, len(original_image) - 5):
        for j in range(5, len(original_image[i]) - 5):
            if output[i][j] > 240 and (
                    original_image[i][j][0] > 150 or original_image[i][j][1] > 150 or original_image[i][j][2] > 150):
                if max_output[i][j] == output[i][j]:
                    if is_red_pixel(original_image, i, j):
                        x_red.append(i)
                        y_red.append(j)
                    else:
                        x_green.append(i)
                        y_green.append(j)

    return tuple(y_red), tuple(x_red), tuple(y_green), tuple(x_green)


def find_tfl_lights(image_path, **kwargs):
    image = processImage(image_path)
    original_image = Image.open(image_path)
    original_image = np.array(original_image)

    kernel_laplician = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

    kernel_big_spots = np.array(
        [[-6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44],
         [-6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44],
         [-6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44],
         [-6.44, -6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, -6.44, -6.44, -6.44],
         [-6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, 7, 7, -6.39, -6.44],
         [-6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, 7, 7, -6.44, -6.44],
         [-6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, 7, 7, -6.44, -6.44],
         [-6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, 7, 7, -6.44, -6.44],
         [-6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, 7, 7, -6.44, -6.44],
         [-6.44, -6.44, -6.44, 7, 7, 7, 7, 7, 7, 7, -6.44, -6.44, -6.44],
         [-6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44],
         [-6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44],
         [-6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44, -6.44]
         ])
    y_red_lap, x_red_lap, y_green_lap, x_green_lap = find_tlf_lights_by_kernel(image, original_image, kernel_laplician)
    y_red_big, x_red_big, y_green_big, x_green_big = find_tlf_lights_by_kernel(image, original_image, kernel_big_spots)
    y_red = y_red_lap + y_red_big
    x_red = x_red_lap + x_red_big
    y_green = y_green_lap + y_green_big
    x_green = x_green_lap + x_green_big


    return tuple(y_red), tuple(x_red), tuple(y_green), tuple(x_green)
