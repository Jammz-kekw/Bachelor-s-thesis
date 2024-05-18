import cv2
import numpy as np


def calculate_histogram(image, channels=[0, 1, 2], histSize=[256], ranges=[0, 256]):
    """
        Calculates the histogram for each channel of the given image. BRG format, tue to cv2

    """
    hist_channels = [cv2.calcHist([image], [i], None, histSize, ranges) for i in channels]
    return hist_channels


def color_correction(source, target):  # source - image to correct; target - correction template
    """
        Prevzatý kód - FIIT STU - Neurónové siete
    """

    """
    Metóda slúži na korekciu farieb pomocou eCDF. Najskôr si vypočítame histogram pre každý kanál `source` a `target` obrázku.
    Následne vypočítame eCDF pre každý histogram všetkých kanálov oboch obrázkov. Nakoniec použijeme lineárnu interpoláciu
    na namapovanie `source` obrázku do `target` obrázku.
    """
    # source_hist = calculate_histogram(source, [[256]], [[0, 256]])
    # target_hist = calculate_histogram(target, [[256]], [[0, 256]])

    source = cv2.imread(source)
    target = cv2.imread(target)

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


if __name__ == '__main__':
    orig_he_1 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1320_orig_he.png"
    gen_he_1 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\1320_ihc_to_he.png"

    orig_he_2 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1082_orig_he.png"
    gen_he_2 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\1082_ihc_to_he.png"

    orig_ihc_1 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\858_orig_ihc.png"
    gen_ihc_1 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\855_he_to_ihc.png"

    orig_ihc_2 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\1792_orig_ihc.png"
    gen_ihc_2 = r"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\1792_he_to_ihc.png"

    inter_he_1 = color_correction(gen_he_1, orig_he_1)
    inter_he_2 = color_correction(gen_he_2, orig_he_1)

    inter_ihc_1 = color_correction(gen_ihc_1, orig_ihc_1)
    inter_ihc_2 = color_correction(gen_ihc_2, orig_ihc_2)

    cv2.imwrite("D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_he_1.png", inter_he_1)
    cv2.imwrite("D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_he_2.png", inter_he_2)

    cv2.imwrite("D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_ihc_1.png", inter_ihc_1)
    cv2.imwrite("D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_ihc_2.png", inter_ihc_2)


