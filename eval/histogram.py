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


def calculate_bhattacharyya_16(image1, image2, patch_size, image1_rgb, image2_rgb, name, image2_normalized_lab):
    patches1 = split_image(image1, patch_size)
    patches2 = split_image(image2, patch_size)
    patches2_normalized = split_image(image2_normalized_lab, patch_size)

    # TODO - toto potom ako u ludi generalizovat do funkcie :)

    bhattacharyya_values_L = []
    bhattacharyya_values_A = []
    bhattacharyya_values_B = []

    bhattacharyya_values_L_norm = []
    bhattacharyya_values_A_norm = []
    bhattacharyya_values_B_norm = []

    for patch1, patch2 in zip(patches1, patches2):
        bhattacharyya_coefficient_L, bhattacharyya_coefficient_A, bhattacharyya_coefficient_B = \
            get_bhattacharyya(patch1, patch2)

        bhattacharyya_values_L.append(bhattacharyya_coefficient_L)
        bhattacharyya_values_A.append(bhattacharyya_coefficient_A)
        bhattacharyya_values_B.append(bhattacharyya_coefficient_B)

    for patch1, patch2_normalized in zip(patches1, patches2_normalized):
        bhattacharyya_coefficient_L_norm, bhattacharyya_coefficient_A_norm, bhattacharyya_coefficient_B_norm = \
            get_bhattacharyya(patch1, patch2_normalized)

        bhattacharyya_values_L_norm.append(bhattacharyya_coefficient_L_norm)
        bhattacharyya_values_A_norm.append(bhattacharyya_coefficient_A_norm)
        bhattacharyya_values_B_norm.append(bhattacharyya_coefficient_B_norm)

    plt.figure(figsize=(16, 9))
    colors = [(0, 'green'), (0.3, 'lime'), (0.5, 'yellow'), (0.6, 'gold'), (0.8, 'orange'), (1, 'red')]

    # Create the custom colormap
    cmap = LinearSegmentedColormap.from_list('excel_heatmap', colors)
    cmap = 'autumn'

    mean_L = np.mean(bhattacharyya_values_L)
    mean_A = np.mean(bhattacharyya_values_A)
    mean_B = np.mean(bhattacharyya_values_B)

    mean_L_norm = np.mean(bhattacharyya_values_L_norm)
    mean_A_norm = np.mean(bhattacharyya_values_A_norm)
    mean_B_norm = np.mean(bhattacharyya_values_B_norm)

    grid_values_L = np.array(bhattacharyya_values_L).reshape((4, 4))
    grid_values_A = np.array(bhattacharyya_values_A).reshape((4, 4))
    grid_values_B = np.array(bhattacharyya_values_B).reshape((4, 4))

    grid_values_L_norm = np.array(bhattacharyya_values_L_norm).reshape((4, 4))
    grid_values_A_norm = np.array(bhattacharyya_values_A_norm).reshape((4, 4))
    grid_values_B_norm = np.array(bhattacharyya_values_B_norm).reshape((4, 4))

    # Plot the grid with the specified colormap
    plt.subplot(3, 3, 4)
    plt.imshow(grid_values_L, cmap=cmap, interpolation='nearest', vmin=0,
               vmax=1)
    plt.title('Bhattacharyya L heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    for i in range(grid_values_L.shape[0]):
        for j in range(grid_values_L.shape[1]):
            value = f'{grid_values_L[i, j]:.5f}'
            plt.text(j, i, value, ha='center', va='center', color='black', rotation=45)

    plt.subplot(3, 3, 5)
    plt.imshow(grid_values_A, cmap=cmap, interpolation='nearest', vmin=0,
               vmax=1)
    plt.title('Bhattacharyya A heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    for i in range(grid_values_A.shape[0]):
        for j in range(grid_values_A.shape[1]):
            value = f'{grid_values_A[i, j]:.5f}'
            plt.text(j, i, value, ha='center', va='center', color='black', rotation=45)

    plt.subplot(3, 3, 6)
    plt.imshow(grid_values_B, cmap=cmap, interpolation='nearest', vmin=0,
               vmax=1)
    plt.title('Bhattacharyya B heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    for i in range(grid_values_B.shape[0]):
        for j in range(grid_values_B.shape[1]):
            value = f'{grid_values_B[i, j]:.5f}'
            plt.text(j, i, value, ha='center', va='center', color='black', rotation=45)

    # Normalized
    plt.subplot(3, 3, 7)
    plt.imshow(grid_values_L_norm, cmap=cmap, interpolation='nearest', vmin=0,
               vmax=1)
    plt.title('Bhattacharyya L normalized heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    for i in range(grid_values_L_norm.shape[0]):
        for j in range(grid_values_L_norm.shape[1]):
            value = f'{grid_values_L_norm[i, j]:.5f}'
            plt.text(j, i, value, ha='center', va='center', color='black', rotation=45)

    plt.subplot(3, 3, 8)
    plt.imshow(grid_values_A_norm, cmap=cmap, interpolation='nearest', vmin=0,
               vmax=1)
    plt.title('Bhattacharyya A normalized heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    for i in range(grid_values_A_norm.shape[0]):
        for j in range(grid_values_A_norm.shape[1]):
            value = f'{grid_values_A_norm[i, j]:.5f}'
            plt.text(j, i, value, ha='center', va='center', color='black', rotation=45)

    plt.subplot(3, 3, 9)
    plt.imshow(grid_values_B_norm, cmap=cmap, interpolation='nearest', vmin=0,
               vmax=1)
    plt.title('Bhattacharyya B normalized heatmap')
    plt.xticks(np.arange(4))  # Set the x-axis ticks
    plt.yticks(np.arange(4))  # Set the y-axis ticks

    for i in range(grid_values_B_norm.shape[0]):
        for j in range(grid_values_B_norm.shape[1]):
            value = f'{grid_values_B_norm[i, j]:.5f}'
            plt.text(j, i, value, ha='center', va='center', color='black', rotation=45)

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Translated')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(image2_normalized_lab, cv2.COLOR_LAB2BGR))
    plt.title('Normalized')
    plt.axis('off')

    plt.figtext(0.23, 0.06, f"Mean Bhattacharyya L - {mean_L}", ha='center')
    plt.figtext(0.51, 0.06, f"Mean Bhattacharyya A - {mean_A}", ha='center')
    plt.figtext(0.78, 0.06, f"Mean Bhattacharyya B - {mean_B}", ha='center')

    plt.figtext(0.23, 0.04, f"Mean Bhattacharyya norm L - {mean_L_norm}", ha='center')
    plt.figtext(0.51, 0.04, f"Mean Bhattacharyya norm A - {mean_A_norm}", ha='center')
    plt.figtext(0.78, 0.04, f"Mean Bhattacharyya norm B - {mean_B_norm}", ha='center')

    plt.suptitle(name)
    plt.savefig(f'D:\\FIIT\\Bachelor-s-thesis\\Dataset\\heatmaps\\run_4x\\{name}.png')
    # plt.show()
    plt.close()

    return mean_L, mean_A, mean_B, mean_L_norm, mean_A_norm, mean_B_norm


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
    plt.savefig(f'D:\\FIIT\\Bachelor-s-thesis\\Dataset\\histograms\\run_4x\\{title}.png')
    # plt.show()
    plt.close()


def calculate(orig_he_folder_path, ihc_to_he_folder_path, tag):
    orig_he_files = os.listdir(orig_he_folder_path)
    ihc_to_he_files = os.listdir(ihc_to_he_folder_path)

    mean_Ls = []
    mean_As = []
    mean_Bs = []

    mean_Ls_norm = []
    mean_As_norm = []
    mean_Bs_norm = []

    for idx, (orig_he_file, ihc_to_he_file) in enumerate(zip(orig_he_files, ihc_to_he_files)):
        if orig_he_file.endswith(".png") and ihc_to_he_file.endswith(".png"):
            gt_image_path = os.path.join(orig_he_folder_path, orig_he_file)
            translated_image_path = os.path.join(ihc_to_he_folder_path, ihc_to_he_file)

            img_gt = cv2.imread(gt_image_path)
            img_translated = cv2.imread(translated_image_path)

            img_translated_normalized = l_channel_normalization(img_translated, img_gt)  # generated | real !
            # img_translated_normalized = lab_channel_normalization(img_translated, img_gt)

            img_translated_normalized_lab = cv2.cvtColor(img_translated_normalized, cv2.COLOR_BGR2LAB)
            img_gt_lab = cv2.cvtColor(img_gt, cv2.COLOR_BGR2LAB)
            img_translated_lab = cv2.cvtColor(img_translated, cv2.COLOR_BGR2LAB)

            name = orig_he_file.split('_')
            name_merged = '_'.join(name[:2])

            # visualize_images(gt_image_path, translated_image_path, name_merged)

            patch_size = 64

            # print(f"{tag} - {name_merged}")
            name_merged = tag + " - " + name_merged

            mean_normalized_mi = calculate_mean_mutual_information(img_gt_lab, img_translated_lab, patch_size)
            # tuto ten LAB - pozor na datovy typ
            # print(f"Mean Normalized Mutual Information - {mean_normalized_mi}")

            mean_L, mean_A, mean_B, mean_L_norm, mean_A_norm, mean_B_norm = calculate_bhattacharyya_16(img_gt_lab, img_translated_lab, 64, img_gt, img_translated, name_merged, img_translated_normalized_lab)

            mean_Ls.append(mean_L)
            mean_As.append(mean_A)
            mean_Bs.append(mean_B)

            mean_Ls_norm.append(mean_L_norm)
            mean_As_norm.append(mean_A_norm)
            mean_Bs_norm.append(mean_L_norm)

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
            # print(f"L bha -  {bhattacharyya_coefficient_L}")
            # print(f"A bha - {bhattacharyya_coefficient_A}")
            # print(f"B bha -  {bhattacharyya_coefficient_B}\n")
            #
            # # the more, the better (more precise)
            # print(f"L cor - {correl_coefficient_L}")
            # print(f"A cor - {correl_coefficient_A}")
            # print(f"B cor - {correl_coefficient_B}\n")

            # TODO - kvalitativne spravit aj heatmapy po normalizacii a porovnat, boli tam nejake problemy s
            # TODO - tym ked sa spravila normalizacia na L, tak uplne odpalilo A a B kanaly do cervenych cisel


            # TODO - kvantitativne cez statisticke medtody na batacharyi, coleracia, mutual

            visualize_lab_histograms(hist_gt_L, hist_gt_A, hist_gt_B,
                                     hist_translated_L, hist_translated_A, hist_translated_B,
                                     img_gt, img_translated, name_merged,
                                     mean_normalized_mi,
                                     bhattacharyya_coefficient_L,
                                     bhattacharyya_coefficient_A,
                                     bhattacharyya_coefficient_B)

    np.save(f'{tag}_L.npy', mean_Ls)
    np.save(f'{tag}_A.npy', mean_As)
    np.save(f'{tag}_B.npy', mean_As)

    np.save(f'{tag}_L_norm.npy', mean_Ls_norm)
    np.save(f'{tag}_A_norm.npy', mean_As_norm)
    np.save(f'{tag}_B_norm.npy', mean_Bs_norm)


def l_channel_normalization(generated_img, ground_truth_img):
    # Convert images to Lab color space
    generated_lab = cv2.cvtColor(generated_img, cv2.COLOR_BGR2LAB)
    ground_truth_lab = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2LAB)

    # Extract Luminance Channel
    generated_L = generated_lab[:, :, 0]
    ground_truth_L = ground_truth_lab[:, :, 0]

    # Normalize L Channels
    generated_mean, generated_std = np.mean(generated_L), np.std(generated_L)
    gt_mean, gt_std = np.mean(ground_truth_L), np.std(ground_truth_L)

    # Apply normalization
    normalized_L = (generated_L - generated_mean) * (gt_std / generated_std) + gt_mean

    # Clip values to valid range [0, 255]
    normalized_L = np.clip(normalized_L, 0, 255)

    # Replace L channel in generated Lab image
    generated_lab[:, :, 0] = normalized_L

    # Convert back to RGB
    normalized_img = cv2.cvtColor(generated_lab, cv2.COLOR_LAB2BGR)

    return normalized_img


def lab_channel_normalization(generated_img, ground_truth_img):
    # Convert images to Lab color space
    generated_lab = cv2.cvtColor(generated_img, cv2.COLOR_BGR2LAB)
    ground_truth_lab = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2LAB)

    # Extract channels
    generated_L, generated_A, generated_B = cv2.split(generated_lab)
    gt_L, gt_A, gt_B = cv2.split(ground_truth_lab)

    # Normalize channels
    for channel in [generated_L, generated_A, generated_B]:
        channel_mean, channel_std = np.mean(channel), np.std(channel)
        gt_mean, gt_std = np.mean(gt_L), np.std(gt_L)
        normalized_channel = (channel - channel_mean) * (gt_std / channel_std) + gt_mean
        # Clip values to valid range [0, 255]
        np.clip(normalized_channel, 0, 255, out=normalized_channel)

    # Merge normalized channels into Lab image
    normalized_lab = cv2.merge([generated_L, generated_A, generated_B])

    # Convert back to RGB
    normalized_img = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    return normalized_img


if __name__ == "__main__":
    orig_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\orig_he'
    ihc_to_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\ihc_to_he'

    orig_ihc_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\orig_ihc'
    he_to_ihc_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\he_to_ihc'

    calculate(orig_he_folder_path, ihc_to_he_folder_path, "HE")
    calculate(orig_ihc_folder_path, he_to_ihc_folder_path, "IHC")


