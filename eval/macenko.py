import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from histomicstk.preprocessing import color_normalization


test_image = "C:\\FIIT\\BP\\Dataset\\sliced\\HE_test\\00026_test_1+_slice_10.jpg"


# Load the image
image = cv2.imread(test_image)

stain_matrix = np.array([[0.644, 0.716, 0.286], [0.092, 0.954, 0.283], [0.310, 0.091, 0.796]])

# Perform deconvolution-based normalization
normalized_image = color_normalization.deconvolution_based_normalization(image, stain_matrix)

# Save the normalized image with a similar pattern
output_path = "C:\\FIIT\\BP\\Dataset\\sliced\\normalized"
slice_name = "normalized.jpg"
slice_path = os.path.join(output_path, slice_name)
cv2.imwrite(slice_path, normalized_image)
cv2.imwrite(output_path + "original.jpg", image)

# Display the original and normalized images using matplotlib
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

# Normalized Image
plt.subplot(1, 2, 2)
plt.imshow(normalized_image)
plt.title('Normalized Image')

plt.show()