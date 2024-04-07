import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from itertools import cycle


def calculate_histogram(image, channels=[0, 1, 2], histSize=[256], ranges=[0, 256]):
    """
    Calculates the histogram for each channel of the given image.
    """
    hist_channels = [cv2.calcHist([image], [i], None, histSize, ranges) for i in channels]
    return hist_channels


def color_correction(source: cv2.Mat, target: cv2.Mat) -> cv2.Mat:
    """
    Metóda slúži na korekciu farieb pomocou eCDF. Najskôr si vypočítame histogram pre každý kanál `source` a `target` obrázku.
    Následne vypočítame eCDF pre každý histogram všetkých kanálov oboch obrázkov. Nakoniec použijeme lineárnu interpoláciu
    na namapovanie `source` obrázku do `target` obrázku.
    """
    # source_hist = calculate_histogram(source, [[256]], [[0, 256]])
    # target_hist = calculate_histogram(target, [[256]], [[0, 256]])

    source_hist = calculate_histogram(source)
    target_hist = calculate_histogram(target)

    source_ecdf = []
    target_ecdf = []

    for i in range(3):
        source_ecdf.append(np.cumsum(source_hist[i]) / np.sum(source_hist[i]))
        target_ecdf.append(np.cumsum(target_hist[i]) / np.sum(target_hist[i]))

    result = np.zeros_like(source)

    for i in range(3):
        lut = np.interp(source_ecdf[i], target_ecdf[i], np.arange(256))
        result[:, :, i] = np.uint8(
            np.interp(source[:, :, i], np.arange(256), lut)
        )

    return result


def compute_values(original_image, generated_image):
    original_patches = split_into_regions(original_image)
    generated_patches = split_into_regions(generated_image)

    l_values_bha = []
    a_values_bha = []
    b_values_bha = []

    l_values_cor = []
    a_values_cor = []
    b_values_cor = []

    for orig_patch, gen_patch in zip(original_patches, generated_patches):
        l_coefficient, a_coefficient, b_coefficient = get_bhattacharyya(orig_patch, gen_patch)

        l_values_bha.append(l_coefficient)
        a_values_bha.append(a_coefficient)
        b_values_bha.append(b_coefficient)

        l_correlation, a_correlation, b_correlation = get_correlation(orig_patch, gen_patch)

        l_values_cor.append(l_correlation)
        a_values_cor.append(a_correlation)
        b_values_cor.append(b_correlation)

    return [l_values_bha, a_values_bha, b_values_bha], [l_values_cor, a_values_cor, b_values_cor]


def visualise(original, generated, normalized, generated_lab, normalized_lab, tag, metric):
    """
            |  orig  |  generated  |  normalized  |
            |--------|-------------|--------------|
            |    x   |      l      |       l      |
            |    x   |      a      |       a      |
            |    x   |      b      |       b      |

            4x3
    """

    plt.close('all')

    if metric == 'bha':
        title = "Bhattacharyya"
        color = 'RdYlGn_r'
        min_range = 0
        max_range = 1
    else:
        title = "Correlation"
        color = 'coolwarm'
        min_range = -1
        max_range = 1

    stain = tag.split('-')[0].strip()
    name = '_'.join([stain, tag.split('- ')[1]])

    gen_l, gen_a, gen_b = generated_lab
    norm_l, norm_a, norm_b = normalized_lab

    grid_gen_l = np.array(gen_l).reshape((4, 4))
    grid_gen_a = np.array(gen_a).reshape((4, 4))
    grid_gen_b = np.array(gen_b).reshape((4, 4))

    grid_norm_l = np.array(norm_l).reshape((4, 4))
    grid_norm_a = np.array(norm_a).reshape((4, 4))
    grid_norm_b = np.array(norm_b).reshape((4, 4))

    plt.figure(figsize=(13, 18))

    # Original image
    plt.subplot(4, 3, 1)
    plt.imshow(original)
    plt.title(f"Pôvodný {stain}")
    plt.axis('off')

    # Generated image
    plt.subplot(4, 3, 2)
    plt.imshow(generated)
    plt.title(f"Generovaný {stain}")
    plt.axis('off')

    # Normalized image
    plt.subplot(4, 3, 3)
    plt.imshow(normalized)
    plt.title(f"Normalizovaný {stain}")
    plt.axis('off')

    # Generated L
    plt.subplot(4, 3, 5)
    plt.imshow(grid_gen_l, cmap=color, interpolation='nearest', vmin=min_range, vmax=max_range)
    plt.title("Generovaný L")
    plt.axis('off')

    for i in range(grid_gen_l.shape[0]):
        for j in range(grid_gen_l.shape[1]):
            value = f'{grid_gen_l[i, j]:.2f}'
            plt.text(j, i, value, ha='center', va='center', color='black')

    # Generated A
    plt.subplot(4, 3, 8)
    plt.imshow(grid_gen_a, cmap=color, interpolation='nearest', vmin=min_range, vmax=max_range)
    plt.title("Generovaný A")
    plt.axis('off')

    for i in range(grid_gen_a.shape[0]):
        for j in range(grid_gen_a.shape[1]):
            value = f'{grid_gen_a[i, j]:.2f}'
            plt.text(j, i, value, ha='center', va='center', color='black')

    # Generated B
    plt.subplot(4, 3, 11)
    plt.imshow(grid_gen_b, cmap=color, interpolation='nearest', vmin=min_range, vmax=max_range)
    plt.title("Generovaný B")
    plt.axis('off')

    for i in range(grid_gen_b.shape[0]):
        for j in range(grid_gen_b.shape[1]):
            value = f'{grid_gen_b[i, j]:.2f}'
            plt.text(j, i, value, ha='center', va='center', color='black')

    # Normalized L
    plt.subplot(4, 3, 6)
    plt.imshow(grid_norm_l, cmap=color, interpolation='nearest', vmin=min_range, vmax=max_range)
    plt.title("Normalizovaný L")
    plt.axis('off')

    for i in range(grid_norm_l.shape[0]):
        for j in range(grid_norm_l.shape[1]):
            value = f'{grid_norm_l[i, j]:.2f}'
            plt.text(j, i, value, ha='center', va='center', color='black')

    # Normalized A
    plt.subplot(4, 3, 9)
    plt.imshow(grid_norm_a, cmap=color, interpolation='nearest', vmin=min_range, vmax=max_range)
    plt.title("Normalizovaný A")
    plt.axis('off')

    for i in range(grid_norm_a.shape[0]):
        for j in range(grid_norm_a.shape[1]):
            value = f'{grid_norm_a[i, j]:.2f}'
            plt.text(j, i, value, ha='center', va='center', color='black')

    # Normalized B
    plt.subplot(4, 3, 12)
    plt.imshow(grid_norm_b, cmap=color, interpolation='nearest', vmin=min_range, vmax=max_range)
    plt.title("Normalizovaný B")
    plt.axis('off')

    for i in range(grid_norm_b.shape[0]):
        for j in range(grid_norm_b.shape[1]):
            value = f'{grid_norm_b[i, j]:.2f}'
            plt.text(j, i, value, ha='center', va='center', color='black')

    # Color bar
    cb_ax = plt.axes((0.3, 0.05, 0.4, 0.02))
    plt.colorbar(ax=plt.gca(), orientation='horizontal', cax=cb_ax)

    # Means on the left side
    plt.figtext(0.23, 0.6, f"Priemer pre generovaný L - {np.mean(gen_l):.4f}", ha='center')
    plt.figtext(0.23, 0.4, f"Priemer pre generovaný A - {np.mean(gen_a):.4f}", ha='center')
    plt.figtext(0.23, 0.2, f"Priemer pre generovaný B - {np.mean(gen_b):.4f}", ha='center')

    plt.figtext(0.23, 0.58, f"Priemer pre normalizovaný L - {np.mean(norm_l):.4f}", ha='center')
    plt.figtext(0.23, 0.38, f"Priemer pre normalizovaný A - {np.mean(norm_a):.4f}", ha='center')
    plt.figtext(0.23, 0.18, f"Priemer pre normalizovaný B - {np.mean(norm_b):.4f}", ha='center')

    # Image
    plt.suptitle(title + " | " + tag)
    plt.savefig(f'D:\\FIIT\\Bachelor-s-thesis\\Dataset\\heatmaps\\run_4x_new\\{name}_{title}.png')
    plt.show()
    plt.close()


def split_into_regions(image):
    patch_size = 64

    patches = []
    height, width = image.shape[:2]

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def l_channel_normalization(original_lab, generated_lab):
    original_l = original_lab[:, :, 0]
    generated_l = generated_lab[:, :, 0]

    original_mean, original_std = np.mean(original_l), np.std(original_l)
    generated_mean, generated_std = np.mean(generated_l), np.std(generated_l)

    normalized_l = (generated_l - generated_mean) * (original_std / generated_std) + original_mean
    normalized_l = np.clip(normalized_l, 0, 255)  # Just to avoid values out of 8-bit space

    generated_lab[:, :, 0] = normalized_l

    normalized_rgb = cv2.cvtColor(generated_lab, cv2.COLOR_LAB2BGR)

    return normalized_rgb


def get_channels(patch):
    l_channel = cv2.calcHist([patch], [0], None, [256], [0, 256])
    a_channel = cv2.calcHist([patch], [1], None, [256], [0, 256])
    b_channel = cv2.calcHist([patch], [2], None, [256], [0, 256])

    l_channel /= l_channel.sum()
    a_channel /= a_channel.sum()
    b_channel /= b_channel.sum()

    return l_channel, a_channel, b_channel


def get_bhattacharyya(original, generated):
    l_orig, a_orig, b_orig = get_channels(original)
    l_gen, a_gen, b_gen = get_channels(generated)

    l_coefficient = cv2.compareHist(l_orig, l_gen, cv2.HISTCMP_BHATTACHARYYA)
    a_coefficient = cv2.compareHist(a_orig, a_gen, cv2.HISTCMP_BHATTACHARYYA)
    b_coefficient = cv2.compareHist(b_orig, b_gen, cv2.HISTCMP_BHATTACHARYYA)

    return l_coefficient, a_coefficient, b_coefficient


def get_correlation(original, generated):
    l_orig, a_orig, b_orig = get_channels(original)
    l_gen, a_gen, b_gen = get_channels(generated)

    l_correlation = cv2.compareHist(l_orig, l_gen, cv2.HISTCMP_CORREL)
    a_correlation = cv2.compareHist(a_orig, a_gen, cv2.HISTCMP_CORREL)
    b_correlation = cv2.compareHist(b_orig, b_gen, cv2.HISTCMP_CORREL)

    return l_correlation, a_correlation, b_correlation


def load_image(name, folder):
    image_path = os.path.join(folder, name)
    image = cv2.imread(image_path)

    return image


def run_pairs(original_files, generated_files, original_path, generated_path, stain_type):
    gen_l_mean_bha = []
    gen_a_mean_bha = []
    gen_b_mean_bha = []

    norm_l_mean_bha = []
    norm_a_mean_bha = []
    norm_b_mean_bha = []

    gen_l_mean_cor = []
    gen_a_mean_cor = []
    gen_b_mean_cor = []

    norm_l_mean_cor = []
    norm_a_mean_cor = []
    norm_b_mean_cor = []

    for _, (original_image, generated_image) in enumerate(zip(original_files, generated_files)):
        """
            1. rgb -> lab
            2. lab normalization on generated image
            3. calculate bhattacharyya and correlation - thus return list of 16 values
               for each channel  
            4. visualise bhattacharyya as 3x4 using green to red gradient
            5. visualise correlation as 3x4 using yellow to red gradient ? # TODO - try this to see the results nech to each other and then compare them
        """

        image_no = original_image.split('_')[0]
        tag = stain_type + " - " + image_no

        original_rgb = load_image(original_image, original_path)
        generated_rgb = load_image(generated_image, generated_path)

        original_lab = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2LAB)
        generated_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_BGR2LAB)

        normalized_rgb = l_channel_normalization(original_lab, generated_lab)
        normalized_lab = cv2.cvtColor(normalized_rgb, cv2.COLOR_BGR2LAB)

        testing_rgb = color_correction(generated_rgb, original_rgb)
        plt.subplot(1, 3, 1)
        plt.imshow(original_rgb)
        plt.axis('off')
        plt.title('Original')

        plt.subplot(1, 3, 2)
        plt.imshow(generated_rgb)
        plt.axis('off')
        plt.title("generated")

        plt.subplot(1, 3, 3)
        plt.imshow(testing_rgb)
        plt.axis('off')
        plt.title('interpolation')

        plt.show()

        # TODO - tuna sprav vizualizaciu pred a po normalizacii pre histogram - staci to spustit len na poslednom HE
        generated_bha, generated_cor = compute_values(original_lab, generated_lab)
        normalized_bha, normalized_cor = compute_values(original_lab, normalized_lab)

        # TODO - pchat to tam teda v RGB alebo v LAB do vizualizacie?
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
        generated_rgb = cv2.cvtColor(generated_rgb, cv2.COLOR_BGR2RGB)
        normalized_rgb = cv2.cvtColor(normalized_rgb, cv2.COLOR_BGR2RGB)

        visualise(original_rgb, generated_rgb, normalized_rgb,
                  generated_bha, normalized_bha, tag, 'bha')

        visualise(original_rgb, generated_rgb, normalized_rgb,
                  generated_cor, normalized_cor, tag, 'cor')

        # Add means
        gen_l_mean_bha.append(np.mean(generated_bha[0]))
        gen_a_mean_bha.append(np.mean(generated_bha[1]))
        gen_b_mean_bha.append(np.mean(generated_bha[2]))

        norm_l_mean_bha.append(np.mean(normalized_bha[0]))
        norm_a_mean_bha.append(np.mean(normalized_bha[1]))
        norm_b_mean_bha.append(np.mean(normalized_bha[2]))

        gen_l_mean_cor.append(np.mean(generated_cor[0]))
        gen_a_mean_cor.append(np.mean(generated_cor[1]))
        gen_b_mean_cor.append(np.mean(generated_cor[2]))

        norm_l_mean_cor.append(np.mean(normalized_cor[0]))
        norm_a_mean_cor.append(np.mean(normalized_cor[1]))
        norm_b_mean_cor.append(np.mean(normalized_cor[2]))

    exit()
    np.save(f'{stain_type}_L_gen_bha.npy', gen_l_mean_bha)
    np.save(f'{stain_type}_A_gen_bha.npy', gen_a_mean_bha)
    np.save(f'{stain_type}_B_gen_bha.npy', gen_b_mean_bha)

    np.save(f'{stain_type}_L_norm_bha.npy', norm_l_mean_bha)
    np.save(f'{stain_type}_A_norm_bha.npy', norm_a_mean_bha)
    np.save(f'{stain_type}_B_norm_bha.npy', norm_b_mean_bha)


    np.save(f'{stain_type}_L_gen_cor.npy', gen_l_mean_cor)
    np.save(f'{stain_type}_A_gen_cor.npy', gen_a_mean_cor)
    np.save(f'{stain_type}_B_gen_cor.npy', gen_b_mean_cor)

    np.save(f'{stain_type}_L_norm_cor.npy', norm_l_mean_cor)
    np.save(f'{stain_type}_A_norm_cor.npy', norm_a_mean_cor)
    np.save(f'{stain_type}_B_norm_cor.npy', norm_b_mean_cor)


def visualise_histograms(original_image, generated_image):
    orig_l, orig_a, orig_b = get_channels(original_image)
    gen_l, gen_a, gen_b = get_channels(generated_image)

    orig_lab = orig_l + orig_a + orig_b
    orig_lab /= orig_lab.sum()

    gen_lab = gen_l + gen_a + gen_b
    gen_lab /= gen_lab.sum()


if __name__ == '__main__':
    orig_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\orig_he'
    ihc_to_he_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\ihc_to_he'

    orig_ihc_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\orig_ihc'
    he_to_ihc_folder_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_4x\\he_to_ihc'

    orig_he_files = os.listdir(orig_he_folder_path)
    ihc_to_he_files = os.listdir(ihc_to_he_folder_path)

    orig_ihc_files = os.listdir(orig_ihc_folder_path)
    he_to_ihc_files = os.listdir(he_to_ihc_folder_path)

    run_pairs(orig_he_files, ihc_to_he_files, orig_he_folder_path, ihc_to_he_folder_path, "HE")

    run_pairs(orig_ihc_files, he_to_ihc_files, orig_ihc_folder_path, he_to_ihc_folder_path, "IHC")

