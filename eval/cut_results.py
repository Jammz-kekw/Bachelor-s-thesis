import cv2
import os

input_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\\results"
output_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut"


def cut_and_save_images(input_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through each input image
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):  # Assuming images are in PNG format, adjust as needed
            # Load the image
            image_path = os.path.join(input_directory, filename)

            # Determine the image name prefix
            image_name_prefix = filename.split("_")[0] + "_" + filename.split("_")[1]

            # Call the function to cut and save images
            cut_and_save_single_image(image_path, image_name_prefix, output_directory)


def cut_and_save_single_image(image_path, image_name_prefix, output_directory):
    image = cv2.imread(image_path)

    # Crop the first image - first row, second column
    first_image = image[:256, 256:512]
    first_image_name = f"{image_name_prefix}_he_to_ihc.png"
    first_image_path = os.path.join(output_directory + "\he_to_ihc", first_image_name)
    cv2.imwrite(first_image_path, first_image)

    # Crop the second image - second row, second column
    second_image = image[256:512, 256:512]
    second_image_name = f"{image_name_prefix}_ihc_to_he.png"
    second_image_path = os.path.join(output_directory + "\ihc_to_he", second_image_name)
    cv2.imwrite(second_image_path, second_image)

    # first row, first column
    orig_he = image[:256, :256]
    orig_he_name = f"{image_name_prefix}_orig_he.png"
    orig_he_path = os.path.join(output_directory + "\orig_he", orig_he_name)
    cv2.imwrite(orig_he_path, orig_he)

    # second row, first column
    orig_ihc = image[256:512, :256]
    orig_ihc_name = f"{image_name_prefix}_orig_ihc.png"
    orig_ihc_path = os.path.join(output_directory + "\orig_ihc", orig_ihc_name)
    cv2.imwrite(orig_ihc_path, orig_ihc)


cut_and_save_images(input_directory, output_directory)
