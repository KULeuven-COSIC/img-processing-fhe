import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
#from skimage.metrics import structural_similarity as ssim

def generate_jpeg_from_csv(input_csv, output_jpeg):
    # Load data from CSV using numpy
    data = np.genfromtxt(input_csv, delimiter=',')

    # Assuming the first column is the x-axis and the remaining columns are y-axes
    x_axis = data[:, 0]
    y_axes = data[:, 1:]

    # Create an array for the image
    image_array = np.zeros((y_axes.shape[0], y_axes.shape[1], 3), dtype=np.uint8)

    # Normalize data to the range [0, 255]
    y_axes_normalized = ((y_axes - np.min(y_axes)) / (np.max(y_axes) - np.min(y_axes)) * 255).astype(np.uint8)

    # Fill the image array with color values
    for i in range(y_axes.shape[1]):
        image_array[:, i, 0] = y_axes_normalized[:, i]
        image_array[:, i, 1] = y_axes_normalized[:, i]
        image_array[:, i, 2] = y_axes_normalized[:, i]

    # Create an Image object from the array
    #image = Image.fromarray(image_array).rotate(-90)
    #im_mirror = ImageOps.mirror(image)

    # Save the image as a JPEG file
    #im_mirror.convert("RGB").save(output_jpeg, "JPEG")

    # Create an Image object from the array
    image = Image.fromarray(image_array)
    #im_mirror = ImageOps.mirror(image)

    # Save the image as a JPEG file
    #im_mirror.convert("RGB").save(output_jpeg, "JPEG")
    image.convert("RGB").save(output_jpeg, "JPEG")

    #return np.array(im_mirror)
    return np.array(image)

    #image.metrics import structural_similarity as ssim
    #ssim_filtered = ssim(np.asarray(conv(img_before,F)), np.asarray(img_after),data_range=256)

def main():
    #img_orig = Image.open(r"boat_image.png").convert('L')
    #image_before = np.asarray(img_orig)
    image_after = generate_jpeg_from_csv('build/output_image.csv', 'output_plot2.jpg')
    #ssim_filtered = ssim(image_before, image_after,data_range=256)
    #print("Image generated SSI: ",ssim_filtered)

main()

