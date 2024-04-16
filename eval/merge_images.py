import cv2
import numpy as np


def merge_images(image1_path, image2_path, name):
    # Read the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Create a white strip as a separator

    white_square = np.ones((256, 256, 3), dtype=np.uint8) * 255

    # Stack the images vertically with the white strip in between
    merged_image = np.hstack((image1, white_square, image2))

    cv2.imwrite(f"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\form\\{name}.png", merged_image)


if __name__ == '__main__':
    orig_he_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1799_orig_he.png"
    gen_he_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\1799_ihc_to_he.png"

    orig_he_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\219_orig_he.png"
    gen_he_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\219_ihc_to_he.png"

    normalized_he_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_he_1.png"
    normalized_he_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_he_2.png"

    merge_images(orig_he_1, orig_he_2, "1_orig_he_1_X_orig_he_2")
    merge_images(orig_he_1, gen_he_1, "2_orig_he_1_X_gen_he_1")
    merge_images(orig_he_2, gen_he_2, "3_orig_he_2_X_gen_he_2")
    merge_images(gen_he_1, gen_he_2, "4_gen_he_1_X_gen_he_2")
    merge_images(orig_he_1, normalized_he_1, "5_orig_he_1_X_norm_he_1")
    merge_images(normalized_he_2, gen_he_2, "6_norm_he_2_X_gen_he_2")
    merge_images(normalized_he_1, orig_he_2, "7_norm_he_1_X_orig_he_2")
    merge_images(normalized_he_1, normalized_he_2, "8_norm_he_1_X_norm_he_2")
    merge_images(gen_he_1, normalized_he_2, "9_gen_he_1_X_norm_he_2")
    merge_images(normalized_he_1, gen_he_2, "10_norm_he_1_X_gen_he_1")

    orig_ihc_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\1799_orig_ihc.png"
    gen_ihc_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\1799_he_to_ihc.png"

    orig_ihc_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\219_orig_ihc.png"
    gen_ihc_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\219_he_to_ihc.png"

    normalized_ihc_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_ihc_1.png"
    normalized_ihc_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_ihc_2.png"

    merge_images(orig_ihc_1, orig_ihc_2, "1_orig_ihc_1_X_orig_ihc_2")
    merge_images(orig_ihc_1, gen_ihc_1, "2_orig_ihc_1_X_gen_ihc_1")
    merge_images(orig_ihc_2, gen_ihc_2, "3_orig_ihc_2_X_gen_ihc_2")
    merge_images(gen_ihc_1, gen_ihc_2, "4_gen_ihc_1_X_gen_ihc_2")
    merge_images(orig_ihc_1, normalized_ihc_1, "5_orig_ihc_1_X_norm_ihc_1")
    merge_images(normalized_ihc_2, gen_ihc_2, "6_norm_ihc_2_X_gen_ihc_2")
    merge_images(normalized_ihc_1, orig_ihc_2, "7_norm_ihc_1_X_orig_ihc_2")
    merge_images(normalized_ihc_1, normalized_ihc_2, "8_norm_ihc_1_X_norm_ihc_2")
    merge_images(gen_ihc_1, normalized_ihc_2, "9_gen_ihc_1_X_norm_ihc_2")
    merge_images(normalized_ihc_1, gen_ihc_2, "10_norm_ihc_1_X_gen_ihc_1")

    orig_he_tr1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1500_orig_he.png"
    gen_he_tr1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\1500_ihc_to_he.png"
    orig_ihc_tr1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\1500_orig_ihc.png"
    gen_ihc_tr1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\1500_he_to_ihc.png"

    orig_he_tr2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\855_orig_he.png"
    gen_he_tr2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\855_ihc_to_he.png"
    orig_ihc_tr2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\855_orig_ihc.png"
    gen_ihc_tr2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\855_he_to_ihc.png"

    orig_he_tr3 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\458_orig_he.png"
    gen_he_tr3 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\458_ihc_to_he.png"
    orig_ihc_tr3 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\458_orig_ihc.png"
    gen_ihc_tr3 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\458_he_to_ihc.png"

    orig_he_tr4 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1119_orig_he.png"
    gen_he_tr4 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\1119_ihc_to_he.png"
    orig_ihc_tr4 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\1119_orig_ihc.png"
    gen_ihc_tr4 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\1119_he_to_ihc.png"

    merge_images(orig_he_tr1, gen_he_tr1, "TR_1")
    merge_images(orig_he_tr2, gen_he_tr2, "TR_2")
    merge_images(orig_he_tr1, gen_he_tr3, "TR_3")
    merge_images(orig_he_tr1, gen_he_tr4, "TR_4")

    merge_images(orig_ihc_tr3, gen_ihc_tr1, "TR_5")
    merge_images(orig_ihc_tr2, gen_ihc_tr2, "TR_6")
    merge_images(orig_ihc_tr3, gen_ihc_tr3, "TR_7")
    merge_images(orig_ihc_tr4, gen_ihc_tr4, "TR_8")

