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


def merge_images_name(image1_path, image2_path, name, text1, text2):
    # Read the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Create a white square
    white_square_size = (256, 256)
    white_square = np.ones((white_square_size[0], white_square_size[1], 3), dtype=np.uint8) * 255

    # Calculate the width of the combined images and square
    total_width = image1.shape[1] + white_square.shape[1] + image2.shape[1]

    # Create a white strip with the correct width
    white_strip = np.ones((40, total_width, 3), dtype=np.uint8) * 255

    # Calculate text sizes and positions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)  # Black color
    text_size1 = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
    text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
    text_x1 = (image1.shape[1] - text_size1[0]) // 2  # Center of the first third
    text_x2 = image1.shape[1] + white_square.shape[1] + (image2.shape[1] - text_size2[0]) // 2  # Center of the last third
    text_y = (white_strip.shape[0] + text_size1[1]) // 2  # Center vertically

    # Write text on the white strip
    cv2.putText(white_strip, text1, (text_x1, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thickness)
    cv2.putText(white_strip, text2, (text_x2, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thickness)

    # Stack the images vertically
    top_images = np.hstack((image1, white_square, image2))

    # Stack the images vertically
    bottom_images = np.vstack((top_images, white_strip))

    # Save the merged image
    cv2.imwrite(f"D:\\FIIT\\Bachelor-s-thesis\\Dataset\\form\\{name}.png", bottom_images)



if __name__ == '__main__':
    orig_he_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1799_orig_he.png"
    gen_he_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\1799_ihc_to_he.png"

    orig_he_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\219_orig_he.png"
    gen_he_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\ihc_to_he\\219_ihc_to_he.png"

    normalized_he_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_he_1.png"
    normalized_he_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_he_2.png"

    merge_images_name(orig_he_1, orig_he_2, "1_orig_he_1_X_orig_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(orig_he_1, gen_he_1, "2_orig_he_1_X_gen_he_1", "Snimok 1", "Snimok 2")
    merge_images_name(orig_he_2, gen_he_2, "3_orig_he_2_X_gen_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(gen_he_1, gen_he_2, "4_gen_he_1_X_gen_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(orig_he_1, normalized_he_1, "5_orig_he_1_X_norm_he_1", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_he_2, gen_he_2, "6_norm_he_2_X_gen_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_he_1, orig_he_2, "7_norm_he_1_X_orig_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_he_1, normalized_he_2, "8_norm_he_1_X_norm_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(gen_he_1, normalized_he_2, "9_gen_he_1_X_norm_he_2", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_he_1, gen_he_2, "10_norm_he_1_X_gen_he_2", "Snimok 1", "Snimok 2")

    orig_ihc_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\1799_orig_ihc.png"
    gen_ihc_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\1799_he_to_ihc.png"

    orig_ihc_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\219_orig_ihc.png"
    gen_ihc_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\he_to_ihc\\219_he_to_ihc.png"

    normalized_ihc_1 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_ihc_1.png"
    normalized_ihc_2 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\normalized\\inter_ihc_2.png"

    merge_images_name(orig_ihc_1, orig_ihc_2, "1_orig_ihc_1_X_orig_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(orig_ihc_1, gen_ihc_1, "2_orig_ihc_1_X_gen_ihc_1", "Snimok 1", "Snimok 2")
    merge_images_name(orig_ihc_2, gen_ihc_2, "3_orig_ihc_2_X_gen_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(gen_ihc_1, gen_ihc_2, "4_gen_ihc_1_X_gen_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(orig_ihc_1, normalized_ihc_1, "5_orig_ihc_1_X_norm_ihc_1", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_ihc_2, gen_ihc_2, "6_norm_ihc_2_X_gen_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_ihc_1, orig_ihc_2, "7_norm_ihc_1_X_orig_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_ihc_1, normalized_ihc_2, "8_norm_ihc_1_X_norm_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(gen_ihc_1, normalized_ihc_2, "9_gen_ihc_1_X_norm_ihc_2", "Snimok 1", "Snimok 2")
    merge_images_name(normalized_ihc_1, gen_ihc_2, "10_norm_ihc_1_X_gen_ihc_2", "Snimok 1", "Snimok 2")

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

    merge_images_name(orig_he_tr1, gen_he_tr1, "TR_1", "Original H&E", "Generovany H&E")
    merge_images_name(orig_he_tr2, gen_he_tr2, "TR_2", "Original H&E", "Generovany H&E")
    merge_images_name(orig_he_tr1, gen_he_tr3, "TR_3", "Original H&E", "Generovany H&E")
    merge_images_name(orig_he_tr1, gen_he_tr4, "TR_4", "Original H&E", "Generovany H&E")

    merge_images_name(orig_ihc_tr3, gen_ihc_tr1, "TR_5", "Original IHC", "Generovany IHC")
    merge_images_name(orig_ihc_tr2, gen_ihc_tr2, "TR_6", "Original IHC", "Generovany IHC")
    merge_images_name(orig_ihc_tr3, gen_ihc_tr3, "TR_7", "Original IHC", "Generovany IHC")
    merge_images_name(orig_ihc_tr4, gen_ihc_tr4, "TR_8", "Original IHC", "Generovany IHC")

    orig_he_3 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_he\\1467_orig_he.png"
    orig_ihc_3 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\orig_ihc\\1467_orig_ihc.png"
    orig_he_4 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_he\\1427_orig_he.png"
    orig_ihc_4 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_ihc\\1427_orig_ihc.png"
    orig_he_5 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_he\\1638_orig_he.png"
    orig_ihc_5 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_ihc\\1638_orig_ihc.png"
    orig_he_6 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_he\\1318_orig_he.png"
    orig_ihc_6 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_ihc\\1318_orig_ihc.png"
    orig_he_7 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_he\\1166_orig_he.png"
    orig_ihc_7 = "D:\\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\orig_ihc\\1166_orig_ihc.png"

    merge_images_name(orig_he_3, orig_ihc_3, "sample_3", "H&E", "IHC")
    merge_images_name(orig_he_4, orig_ihc_4, "sample_4", "H&E", "IHC")
    merge_images_name(orig_he_5, orig_ihc_5, "sample_5", "H&E", "IHC")
    merge_images_name(orig_he_6, orig_ihc_6, "sample_6", "H&E", "IHC")
    merge_images_name(orig_he_7, orig_ihc_7, "sample_7", "H&E", "IHC")
