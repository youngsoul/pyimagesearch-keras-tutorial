# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

def generate_augmented_image(image_path, output_dir, total_aug_images, prefix):
    # load the input image, convert itto a numpy array and then reshape it to have an extra dimension
    logging.debug(f"loading example image: {image_path}")
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # construct the image generator for the data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    total = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # construct the actual python generator
    logging.debug("generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix=prefix, save_format="jpg")

    for image in imageGen:
        # increment counter
        total += 1

        # if we have reach the specifiedc number of examples, break from the loop
        if total == total_aug_images:
            break

if __name__ == '__main__':

    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    ap.add_argument("-o", "--output", required=True, help="path to the output directory to store augmentations examples")
    ap.add_argument("-t", "--total", type=int, default=100, help="# of training samples to generate")
    ap.add_argument("-p", "--prefix", type=str, default="image", help="prefix to add to all generated images")

    args = vars(ap.parse_args())

    generate_augmented_image(args['image'], args['output'], args['total'], args['prefix'])

