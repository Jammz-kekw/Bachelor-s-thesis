## TODO -  using opencv slide 1024x1024 into 16 256x256 images
## consider overlaping edges by 32px - 12,5% - merging the images after is debatable
## image size needs to be 256x256 due to the input data requirements

import cv2
import os

input_dir = "C:\FIIT\BP\Dataset\\groundtruth"
output_dir = "C:\FIIT\BP\Dataset\sliced\IHC_groundtruth"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def slice_and_save(image_path, output_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    slice_size = 256

    count = 1
    for i in range(0, height, slice_size):
        for j in range(0, width, slice_size):
            slice_img = img[i:i+slice_size, j:j+slice_size]

            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            slice_name = f"{base_filename}_slice_{count}.jpg"
            slice_path = os.path.join(output_path, slice_name)
            cv2.imwrite(slice_path, slice_img)

            count += 1


for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    slice_and_save(input_path, output_dir)
