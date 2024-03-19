import os
import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def split_image(image, patch_size):
    patches = []
    height, width = image.shape[:2]

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def visualise_split(patch):
    i = 1
    for each in patch:
        plt.subplot(4, 4, i)
        plt.imshow(cv2.cvtColor(each, cv2.COLOR_LAB2RGB))
        i += 1
    plt.show()


def calculate_mean_mutual_information(image1, image2, patch_size):
    patches1 = split_image(image1, patch_size)
    patches2 = split_image(image2, patch_size)

    # visualise_split(patches1)
    # visualise_split(patches2)

    mutual_information_values = []

    for patch1, patch2 in zip(patches1, patches2):
        flat_patch1 = patch1.flatten()
        flat_patch2 = patch2.flatten()
        mutual_info = normalized_mutual_information(flat_patch1, flat_patch2)
        mutual_information_values.append(mutual_info)

    mean_mutual_information = np.mean(mutual_information_values)

    return mean_mutual_information


def get_bhattacharyya(image1, image2):
    hist_gt_L = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist_gt_A = cv2.calcHist([image1], [1], None, [256], [0, 256])
    hist_gt_B = cv2.calcHist([image1], [2], None, [256], [0, 256])

    hist_translated_L = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist_translated_A = cv2.calcHist([image2], [1], None, [256], [0, 256])
    hist_translated_B = cv2.calcHist([image2], [2], None, [256], [0, 256])

    hist_gt_L /= hist_gt_L.sum()
    hist_gt_A /= hist_gt_A.sum()
    hist_gt_B /= hist_gt_B.sum()

    hist_translated_L /= hist_translated_L.sum()
    hist_translated_A /= hist_translated_A.sum()
    hist_translated_B /= hist_translated_B.sum()

    bhattacharyya_coefficient_L = cv2.compareHist(hist_gt_L, hist_translated_L, cv2.HISTCMP_BHATTACHARYYA)
    bhattacharyya_coefficient_A = cv2.compareHist(hist_gt_A, hist_translated_A, cv2.HISTCMP_BHATTACHARYYA)
    bhattacharyya_coefficient_B = cv2.compareHist(hist_gt_B, hist_translated_B, cv2.HISTCMP_BHATTACHARYYA)

    return bhattacharyya_coefficient_L, bhattacharyya_coefficient_A, bhattacharyya_coefficient_B


def calculate_bhattacharyya_16(image1, image2, patch_size, image1_rgb, image2_rgb, name):
    patches1 = split_image(image1, patch_size)
    patches2 = split_image(image2, patch_size)

    bhattacharyya_values_L = []
    bhattacharyya_values_A = []
    bhattacharyya_values_B = []

    for patch1, patch2 in zip(patches1, patches2):
        bhattacharyya_coefficient_L, bhattacharyya_coefficient_A, bhattacharyya_coefficient_B = \
            get_bhattacharyya(patch1, patch2)

        bhattacharyya_values_L.append(bhattacharyya_coefficient_L)
        bhattacharyya_values_A.append(bhattacharyya_coefficient_A)
        bhattacharyya_values_B.append(bhattacharyya_coefficient_B)

    i = 1
    plt.figure(figsize=(16, 9))
    colors = [(0, 'green'), (0.3, 'lime'), (0.5, 'yellow'), (0.6, 'gold'), (0.8, 'orange'), (1, 'red')]

    # Create the custom colormap
    cmap = LinearSegmentedColormap.from_list('excel_heatmap', colors)

    mean_L = np.mean(bhattacharyya_values_L)
    mean_A = np.mean(bhattacharyya_values_A)
    mean_B = np.mean(bhattacharyya_values_B)

    grid_values = np.array(bhattacharyya_values_L).reshape((4, 4))

    # Plot the grid with the specified colormap
    plt.subplot(1, 3, 2)
    plt.imshow(grid_values, cmap=cmap, interpolation='nearest', vmin=min(bhattacharyya_values_L), vmax=max(bhattacharyya_values_L))
    plt.title('Bhattacharyya heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Original - Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Original - Translated')
    plt.axis('off')

    plt.figtext(0.23, 0.15, f"Mean Bhattacharyya L - {mean_L}", ha='center')
    plt.figtext(0.51, 0.15, f"Mean Bhattacharyya A - {mean_A}", ha='center')
    plt.figtext(0.78, 0.15, f"Mean Bhattacharyya B - {mean_B}", ha='center')

    plt.suptitle(name)
    plt.show()






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


def visualize_lab_histograms(hist_gt_L, hist_gt_A, hist_gt_B, hist_translated_L, hist_translated_A, hist_translated_B,
                             img_gt, img_translated, title, mean_normalized_mi, bhattacharyya_coefficient_L,
                             bhattacharyya_coefficient_A, bhattacharyya_coefficient_B):

    hist_gt_merged = hist_gt_L + hist_gt_A + hist_gt_B
    hist_translated_merged = hist_translated_L + hist_translated_A + hist_translated_B

    hist_gt_merged /= hist_gt_merged.sum()
    hist_translated_merged /= hist_translated_merged.sum()

    plt.figure(figsize=(15, 10))

    # Plot for original images
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.title('Original - Ground Truth')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB))
    plt.title('Original - Translated')
    plt.axis('off')

    # Plot for L channel
    plt.subplot(2, 3, 4)
    plt.plot(hist_gt_L, color='r', label='Ground Truth')
    plt.plot(hist_translated_L, color='b', linestyle='--', label='Translated')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram - L Channel')
    plt.legend()

    # Plot for A channel
    plt.subplot(2, 3, 5)
    plt.plot(hist_gt_A, color='r', label='Ground Truth')
    plt.plot(hist_translated_A, color='b', linestyle='--', label='Translated')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram - A Channel')
    plt.legend()

    # Plot for B channel
    plt.subplot(2, 3, 6)
    plt.plot(hist_gt_B, color='r', label='Ground Truth')
    plt.plot(hist_translated_B, color='b', linestyle='--', label='Translated')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram - B Channel')
    plt.legend()

    # Plot for LAB as a whole
    plt.subplot(2, 3, 2)
    plt.plot(hist_gt_merged, color='r', label='Ground Truth')
    plt.plot(hist_translated_merged, color='b', linestyle='--', label='Translated')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram - LAB comparison')
    plt.legend()

    plt.subplots_adjust(bottom=0.14)

    plt.figtext(0.51, 0.03, f"Mean normalized mutual information - {mean_normalized_mi}", ha='center')
    plt.figtext(0.24, 0.06, f"Bhattacharyya L - {bhattacharyya_coefficient_L}", ha='center')
    plt.figtext(0.51, 0.06, f"Bhattacharyya A - {bhattacharyya_coefficient_A}", ha='center')
    plt.figtext(0.79, 0.06, f"Bhattacharyya B - {bhattacharyya_coefficient_B}", ha='center')

    plt.suptitle(title)
    # plt.tight_layout()
    plt.savefig(f'D:\\FIIT\\Bachelor-s-thesis\\Dataset\\histograms\\{title}.png')
    plt.show()


def calculate(orig_he_folder_path, ihc_to_he_folder_path, tag):
    orig_he_files = os.listdir(orig_he_folder_path)
    ihc_to_he_files = os.listdir(ihc_to_he_folder_path)

    for idx, (orig_he_file, ihc_to_he_file) in enumerate(zip(orig_he_files, ihc_to_he_files)):
        if orig_he_file.endswith(".png") and ihc_to_he_file.endswith(".png"):
            gt_image_path = os.path.join(orig_he_folder_path, orig_he_file)
            translated_image_path = os.path.join(ihc_to_he_folder_path, ihc_to_he_file)

            img_gt = cv2.imread(gt_image_path)
            img_translated = cv2.imread(translated_image_path)

            img_gt_lab = cv2.cvtColor(img_gt, cv2.COLOR_BGR2LAB)
            img_translated_lab = cv2.cvtColor(img_translated, cv2.COLOR_BGR2LAB)

            name = orig_he_file.split('_')
            name_merged = '_'.join(name[:2])

            # visualize_images(gt_image_path, translated_image_path, name_merged)

            patch_size = 64

            print(f"{tag} - {name_merged}")
            name_merged = tag + " - " + name_merged

            mean_normalized_mi = calculate_mean_mutual_information(img_gt_lab, img_translated_lab, patch_size)
            # tuto ten LAB - pozor na datovy typ
            print(f"Mean Normalized Mutual Information - {mean_normalized_mi}")

            calculate_bhattacharyya_16(img_gt_lab, img_translated_lab, 64, img_gt, img_translated, name_merged)

            hist_gt_L = cv2.calcHist([img_gt_lab], [0], None, [256], [0, 256])
            hist_gt_A = cv2.calcHist([img_gt_lab], [1], None, [256], [0, 256])
            hist_gt_B = cv2.calcHist([img_gt_lab], [2], None, [256], [0, 256])

            hist_translated_L = cv2.calcHist([img_translated_lab], [0], None, [256], [0, 256])
            hist_translated_A = cv2.calcHist([img_translated_lab], [1], None, [256], [0, 256])
            hist_translated_B = cv2.calcHist([img_translated_lab], [2], None, [256], [0, 256])

            # Normalize histograms
            hist_gt_L /= hist_gt_L.sum()
            hist_gt_A /= hist_gt_A.sum()
            hist_gt_B /= hist_gt_B.sum()

            hist_translated_L /= hist_translated_L.sum()
            hist_translated_A /= hist_translated_A.sum()
            hist_translated_B /= hist_translated_B.sum()

            # Calculate Bhattacharyya coefficients
            bhattacharyya_coefficient_L = cv2.compareHist(hist_gt_L, hist_translated_L, cv2.HISTCMP_BHATTACHARYYA)
            bhattacharyya_coefficient_A = cv2.compareHist(hist_gt_A, hist_translated_A, cv2.HISTCMP_BHATTACHARYYA)
            bhattacharyya_coefficient_B = cv2.compareHist(hist_gt_B, hist_translated_B, cv2.HISTCMP_BHATTACHARYYA)

            correl_coefficient_L = cv2.compareHist(hist_gt_L, hist_translated_L, cv2.HISTCMP_CORREL)
            correl_coefficient_A = cv2.compareHist(hist_gt_A, hist_translated_A, cv2.HISTCMP_CORREL)
            correl_coefficient_B = cv2.compareHist(hist_gt_B, hist_translated_B, cv2.HISTCMP_CORREL)

            # the less, the better (more precise)
            # TODO - normalizacia LAB celkovo
            print(f"L bha -  {bhattacharyya_coefficient_L}")
            print(f"A bha - {bhattacharyya_coefficient_A}")
            print(f"B bha -  {bhattacharyya_coefficient_B}\n")

            # the more, the better (more precise)
            print(f"L cor - {correl_coefficient_L}")
            print(f"A cor - {correl_coefficient_A}")
            print(f"B cor - {correl_coefficient_B}\n")

            # TODO - kvalitativne cez color scaling na patche - heatmap
            # takze treba zobrat jednotlive close-upy a

            # TODO - kvantitativne cez statisticke medtody na batacharyi, coleracia, mutual

            visualize_lab_histograms(hist_gt_L, hist_gt_A, hist_gt_B,
                                     hist_translated_L, hist_translated_A, hist_translated_B,
                                     img_gt, img_translated, name_merged,
                                     mean_normalized_mi,
                                     bhattacharyya_coefficient_L,
                                     bhattacharyya_coefficient_A,
                                     bhattacharyya_coefficient_B)


if __name__ == "__main__":
    orig_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\orig_he'
    ihc_to_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\ihc_to_he'

    orig_ihc_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\orig_ihc'
    he_to_ihc_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\he_to_ihc'

    # 1646, 1640, 1605, 1602, 1496, 1491, 1450

    calculate(orig_he_folder_path, ihc_to_he_folder_path, "HE")
    calculate(orig_ihc_folder_path, he_to_ihc_folder_path, "IHC")


