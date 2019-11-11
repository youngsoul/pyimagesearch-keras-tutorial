from pathlib import Path
from generate_images import generate_augmented_image
import argparse

if __name__ == '__main__':
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--source-dir", required=False, default='./animals',  help="path to the input image")
    ap.add_argument("-o", "--output-dir", required=False, default='./animals_aug', help="path to the output directory to store augmentations examples")
    ap.add_argument("-t", "--num-aug-per-image", type=int, default=10, help="# of training samples to generate")

    args = vars(ap.parse_args())

    source_dir = Path(args['source_dir'])
    for file in source_dir.glob("**/*"):
        if file.is_dir():
            continue
        print(file)
        output_dir = str(file.relative_to(".")).split("/")
        image_label = output_dir[1]
        base_file_name = str(file).split("/")[-1].split(".")[0]
        output_dir[0] = args['output_dir']
        output_dir = "/".join(output_dir[:2])
        generate_augmented_image(file, output_dir, 10, f"aug_{base_file_name}_")
