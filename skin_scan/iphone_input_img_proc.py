import os
import numpy as np
from PIL import Image

def process_image(image, width, height):
    img_converted = image.convert("RGB")
    img_converted_resized = img_converted.resize((width, height))
    img_array = np.array(img_converted_resized)
    return img_array

def process_input_image(path: os.path) -> np.array:
    # Open the HEIC image
    img = Image.open('../raw_data/input_photos/IMG_1521.HEIC')

    # Rotate 90 degrees counterclockwise
    img_rotated = img.rotate(90, expand=True)

    # Resize Image to
    img_rotated.thumbnail((600, 450), Image.Resampling.LANCZOS)

    # Strip metadata (by creating a new image)
    img_clean = Image.new(img_rotated.mode, img_rotated.size)
    img_clean.putdata(list(img_rotated.getdata()))

    # apply the same preprocessing that the dataset gets
    img_array = process_image(img_clean, 96, 96)

    #return the np.array of the image
    return img_array

