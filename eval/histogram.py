import os
import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information
import matplotlib.pyplot as plt


def split_image(image, patch_size):
    patches = []
    height, width = image.shape[:2]

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches


def calculate_mean_mutual_information(image1, image2, patch_size):
    patches1 = split_image(image1, patch_size)
    patches2 = split_image(image2, patch_size)

    mutual_information_values = []

    for patch1, patch2 in zip(patches1, patches2):
        flat_patch1 = patch1.flatten()
        flat_patch2 = patch2.flatten()
        mutual_info = normalized_mutual_information(flat_patch1, flat_patch2)
        mutual_information_values.append(mutual_info)

    mean_mutual_information = np.mean(mutual_information_values)

    return mean_mutual_information


def visualize_images(image_gt, image_translated, run_no):
    img_gt = cv2.imread(image_gt)
    img_translated = cv2.imread(image_translated)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.title(f'{run_no} - original')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB))
    plt.title(f'{run_no} - translated')

    plt.show()


if __name__ == "__main__":
    orig_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\orig_he'
    ihc_to_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\ihc_to_he'

    orig_he_files = os.listdir(orig_he_folder_path)
    ihc_to_he_files = os.listdir(ihc_to_he_folder_path)

    for idx, (orig_he_file, ihc_to_he_file) in enumerate(zip(orig_he_files, ihc_to_he_files)):
        if orig_he_file.endswith(".png") and ihc_to_he_file.endswith(".png"):  # Adjust file extensions if needed
            gt_image_path = os.path.join(orig_he_folder_path, orig_he_file)
            translated_image_path = os.path.join(ihc_to_he_folder_path, ihc_to_he_file)

            img_gt = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
            img_translated = cv2.imread(translated_image_path, cv2.IMREAD_GRAYSCALE)

            name = orig_he_file.split('_')
            name_merged = '_'.join(name[:2])

            visualize_images(gt_image_path, translated_image_path, name_merged)

            patch_size = 64

            print(f"{name_merged}")

            mean_normalized_mi = calculate_mean_mutual_information(img_gt, img_translated, patch_size)
            print(f"Mean Normalized Mutual Information - {mean_normalized_mi}")

            hist_gt = cv2.calcHist([img_gt], [0], None, [256], [0, 256])
            hist_translated = cv2.calcHist([img_translated], [0], None, [256], [0, 256])

            hist_gt /= hist_gt.sum()
            hist_translated /= hist_translated.sum()

            bhattacharyya_coefficient = cv2.compareHist(hist_gt, hist_translated, cv2.HISTCMP_BHATTACHARYYA)
            print(f"Bhattacharyya Coefficient - {bhattacharyya_coefficient}\n")
