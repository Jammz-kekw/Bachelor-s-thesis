import cv2
import os


def cut_and_save_images(input_directory, output_directory):
    """
        Used as a driver code to iterate through directory with images

    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".png") or filename.endswith('.jpg'):
            image_path = os.path.join(input_directory, filename)

            # image_name_prefix = filename.split("_")[0] + "_" + filename.split("_")[1]
            image_name_prefix = filename.split(".")[0]

            cut_and_save_single_image(image_path, image_name_prefix, output_directory)


def rename(input_directory):
    """
        Used to rename target images, making them easier to work with

    """

    files = os.listdir(input_directory)

    for filename in files:
        parts = filename.split('_')
        number = parts[2]
        new_filename = f"{number}.png"

        os.rename(os.path.join(input_directory, filename), os.path.join(input_directory, new_filename))


def cut_and_save_single_image(image_path, image_name_prefix, output_directory):
    """
        Used to cut GAN generated and ground-truth images downloaded from wandb.

    """

    image = cv2.imread(image_path)

    # first row, second column
    first_image = image[:256, 256:512]
    first_image_name = f"{image_name_prefix}_he_to_ihc.png"
    first_image_path = os.path.join(output_directory + r"\he_to_ihc", first_image_name)
    cv2.imwrite(first_image_path, first_image)

    # first row, first column
    orig_he = image[:256, :256]
    orig_he_name = f"{image_name_prefix}_orig_he.png"
    orig_he_path = os.path.join(output_directory + r"\orig_he", orig_he_name)
    cv2.imwrite(orig_he_path, orig_he)

    # second row, first column
    orig_ihc = image[256:512, :256]
    orig_ihc_name = f"{image_name_prefix}_orig_ihc.png"
    orig_ihc_path = os.path.join(output_directory + r"\orig_ihc", orig_ihc_name)
    cv2.imwrite(orig_ihc_path, orig_ihc)

    # second row, second column
    second_image = image[256:512, 256:512]
    second_image_name = f"{image_name_prefix}_ihc_to_he.png"
    second_image_path = os.path.join(output_directory + r"\ihc_to_he", second_image_name)
    cv2.imwrite(second_image_path, second_image)


if __name__ == '__main__':
    input_directory = r"D:\FIIT\Bachelor-s-thesis\Dataset\\results\\run_4x"
    output_directory = r"D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x"

    # rename(input_directory)
    cut_and_save_images(input_directory, output_directory)
