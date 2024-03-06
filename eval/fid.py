# import os
# import random
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
#
# fid_scores = []
# interpretations = []
#
#
# def load_and_preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (299, 299))
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image
#
#
# def extract_features(image_path, model):
#     image = load_and_preprocess_image(image_path)
#     features = model.predict(image)
#     return features.flatten()
#
#
# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
#     term1 = np.sum((mu1 - mu2)**2)
#     outer_product = np.outer(mu1 - mu2, mu1 - mu2)
#     term2 = np.trace(sigma1 + sigma2 - 2 * np.sqrt(np.outer(sigma1, sigma2)))
#     return term1 + term2
#
#
# def calculate_fid(real_features, generated_features):
#     mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
#     mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
#     fid = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
#     return fid
#
#
# def interpret_fid_score(fid_score):
#     if fid_score < 0.05:
#         return "Excellent"
#     elif 0.05 <= fid_score < 0.1:
#         return "Very Good"
#     elif 0.1 <= fid_score < 0.2:
#         return "Good"
#     elif 0.2 <= fid_score < 0.3:
#         return "Fair"
#     else:
#         return "Poor"
#
#
# def select_random_images(directory, sample_size=50):
#     all_files = os.listdir(directory)
#     image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
#
#     if len(image_files) < sample_size:
#         raise ValueError("Not enough images in the directory.")
#
#     selected_images = random.sample(image_files, sample_size)
#     return [os.path.join(directory, image) for image in selected_images]
#
#
# # Load the pre-trained Inception v3 model
# inception_model = InceptionV3(weights='imagenet', include_top=False)
#
# # Specify the paths to the real and generated images directory
# real_images_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\sliced\HE_test"
# generated_images_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut"
#
# # Get a random sample of images from each directory
# real_sample = select_random_images(real_images_directory, sample_size=50)
# generated_sample = select_random_images(generated_images_directory, sample_size=1)
#
# # Iterate over the pairs of real and generated images, calculate FID, and store the results
# for real_image_path, generated_image_path in zip(real_sample, generated_sample):
#     real_features = extract_features(real_image_path, inception_model)
#     generated_features = extract_features(generated_image_path, inception_model)
#
#     fid_score = calculate_fid(real_features, generated_features)
#     fid_scores.append(fid_score)
#
#     interpretation = interpret_fid_score(fid_score)
#     interpretations.append(interpretation)
#
# # Convert interpretations to colors
# interpretation_colors = {'Excellent': 'green', 'Very Good': 'blue', 'Good': 'yellow', 'Fair': 'orange', 'Poor': 'red'}
# colors = [interpretation_colors[interpretation] for interpretation in interpretations]
#
# # Plot the FID scores
# plt.scatter(range(len(fid_scores)), fid_scores, c=colors)
# plt.xlabel('Sample Index')
# plt.ylabel('FID Score')
# plt.title('FID Scores Visualization')
#
# # Create custom legend with colored dots and interpretation
# legend_labels = [f'{interpretation[0]}' for interpretation in interpretation_colors.items()]
# legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in interpretation_colors.values()]
# plt.legend(legend_handles, legend_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='center left')
#
# plt.show()

## **********************************************************************************************************************************************************************

# import os
# import random
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
#
#
# fid_scores = []
# interpretations = []
#
#
# def load_and_preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (299, 299))
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image
#
#
# def extract_features(image_path, model):
#     image = load_and_preprocess_image(image_path)
#     features = model.predict(image)
#     return features.flatten()
#
#
# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
#     term1 = np.sum((mu1 - mu2) ** 2)
#     outer_product = np.outer(mu1 - mu2, mu1 - mu2)
#     term2 = np.trace(sigma1 + sigma2 - 2 * np.sqrt(np.outer(sigma1, sigma2)))
#     return term1 + term2
#
#
# def calculate_fid(real_features, generated_features):
#     mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
#     mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
#     fid = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
#     return fid
#
#
# def interpret_fid_score(fid_score):
#     if fid_score < 0.05:
#         return "Excellent"
#     elif 0.05 <= fid_score < 0.1:
#         return "Very Good"
#     elif 0.1 <= fid_score < 0.2:
#         return "Good"
#     elif 0.2 <= fid_score < 0.3:
#         return "Fair"
#     else:
#         return "Poor"
#
#
# def select_random_images(directory, sample_size=50):
#     all_files = os.listdir(directory)
#     image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
#
#     if len(image_files) < sample_size:
#         raise ValueError("Not enough images in the directory.")
#
#     selected_images = random.sample(image_files, sample_size)
#     return [os.path.join(directory, image) for image in selected_images]
#
#
# # Load the pre-trained Inception v3 model
# inception_model = InceptionV3(weights='imagenet', include_top=False)
#
# # Specify the paths to the real and generated images directory
# real_images_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\sliced\HE_test"
# generated_images_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut"
#
# # Get a random sample of images from the real directory
# real_sample = select_random_images(real_images_directory, sample_size=500)
#
# # Select one image from the generated directory
# # generated_image = select_random_images(generated_images_directory, sample_size=1)[0]
# generated_image = "D:\FIIT\Bachelor-s-thesis\Dataset\\results\\fiit_logo.png"
#
# # Iterate over the pairs of real and the single generated image, calculate FID, and store the result
# generated_features = extract_features(generated_image, inception_model)
#
# for real_image_path in real_sample:
#     real_features = extract_features(real_image_path, inception_model)
#
#     fid_score = calculate_fid(real_features, generated_features)
#     fid_scores.append(fid_score)
#
#     interpretation = interpret_fid_score(fid_score)
#     interpretations.append(interpretation)
#
# num_rows, num_cols = 1, 2
#
# # Create a subplot with 1 row and 2 columns
# fig, (ax1, ax2) = plt.subplots(num_rows, num_cols, figsize=(12, 4))
#
# # Display the generated image in the first subplot
# generated_image_data = cv2.imread(generated_image)
# ax1.imshow(cv2.cvtColor(generated_image_data, cv2.COLOR_BGR2RGB))
# ax1.axis('off')
# ax1.set_title(f"Generated Image: {os.path.basename(generated_image)}")
#
# # Display the scatter plot in the second subplot
# interpretation_colors = {'Excellent': 'green', 'Very Good': 'blue', 'Good': 'yellow', 'Fair': 'orange', 'Poor': 'red'}
# colors = [interpretation_colors[interpretation] for interpretation in interpretations]
#
# ax2.scatter(range(len(fid_scores)), fid_scores, c=colors)
# ax2.set_xlabel('Amount of real images sample')
# ax2.set_ylabel('FID Score')
# ax2.set_title(f"FID Score - {os.path.basename(generated_image)}")
#
# # Create custom legend with colored dots and interpretation
# legend_labels = [f'{interpretation[0]}' for interpretation in interpretation_colors.items()]
# legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
#                   interpretation_colors.values()]
# ax2.legend(legend_handles, legend_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='center left')
#
# plt.tight_layout()
# plt.show()
 ## **************************
# import os
# import cv2
# import numpy as np
# from scipy.linalg import sqrtm
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input
#
#
# def load_images_from_directory(directory, num_images=10000):
#     images = []
#     filenames = os.listdir(directory)
#     selected_filenames = np.random.choice(filenames, num_images, replace=False)
#
#     for filename in selected_filenames:
#         if filename.endswith(".png") or filename.endswith(".jpg"):
#             img_path = os.path.join(directory, filename)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (299, 299))
#             images.append(img)
#
#     return np.array(images)
#
#
# def preprocess_images(images):
#     images = images.astype('float32')
#     images = preprocess_input(images)
#     return images
#
#
# def calculate_fid(real_images, generated_image):
#     model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
#
#     real_features = model.predict(preprocess_images(real_images))
#     generated_features = model.predict(preprocess_images(np.array([generated_image])))
#
#     mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
#     mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
#
#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)
#
#     epsilon = 1e-6
#     sigma1 += epsilon * np.eye(sigma1.shape[0])
#     sigma2 += epsilon * np.eye(sigma2.shape[0])
#
#     eigval1, eigvec1 = np.linalg.eigh(sigma1)
#     eigval2, eigvec2 = np.linalg.eigh(sigma2)
#
#     sqrt_eigval1 = np.sqrt(np.abs(eigval1))
#     sqrt_eigval2 = np.sqrt(np.abs(eigval2))
#
#     if not np.isfinite(sqrt_eigval1).all() or not np.isfinite(sqrt_eigval2).all():
#         fid = np.inf
#     else:
#         fid = np.real(np.sum((sqrt_eigval1 - sqrt_eigval2) ** 2) +
#                       np.trace(np.dot(np.dot(eigvec1, np.diag(sqrt_eigval1)), eigvec1.T)))
#
#     return fid
#
#
# real_images_directory = 'D:\FIIT\Bachelor-s-thesis\Dataset\sliced\HE_test'
# generated_image_path = 'D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_1_ihc_to_he.png'
#
# real_images = load_images_from_directory(real_images_directory, num_images=2000)
#
# generated_image = cv2.imread(generated_image_path)
# generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
# generated_image = cv2.resize(generated_image, (299, 299))
#
# fid_score = calculate_fid(real_images, generated_image)
# print("FID Score:", fid_score)

## ****************************************************************************************************************************************

import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2


def load_and_preprocess_real_images(num_images):
    # Implement this function to load and preprocess your real images
    # Example: Read images from a directory and resize to (299, 299)
    real_images = []
    real_images_directory = "D:\FIIT\Bachelor-s-thesis\Dataset\sliced\HE_test"
    for filename in os.listdir(real_images_directory)[:num_images]:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(real_images_directory, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (299, 299))
            real_images.append(image)
    return np.array(real_images)


def load_and_preprocess_generated_image():
    # Implement this function to load and preprocess your generated image
    # Example: Read a single generated image and resize to (299, 299)
    generated_image_path = "D:\FIIT\Bachelor-s-thesis\Dataset\\results_cut\\run_10_ihc_to_he.png"
    generated_image = cv2.imread(generated_image_path)
    generated_image = cv2.resize(generated_image, (299, 299))
    return generated_image


def preprocess_images(images):
    # Implement preprocessing as needed
    return preprocess_input(images)


def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Add epsilon to the diagonal of covariance matrices
    epsilon = 1e-6
    sigma_real += epsilon * np.eye(sigma_real.shape[0])

    if np.isscalar(sigma_generated):
        sigma_generated += epsilon
    else:
        sigma_generated += epsilon * np.eye(sigma_generated.shape[0])

    # Rest of the FID calculation remains unchanged
    eigval_real, eigvec_real = np.linalg.eigh(sigma_real)
    eigval_generated, eigvec_generated = np.linalg.eigh(sigma_generated)

    sqrt_eigval_real = np.sqrt(np.abs(eigval_real))
    sqrt_eigval_generated = np.sqrt(np.abs(eigval_generated))

    fid = np.real(np.sum((mu_real - mu_generated) ** 2) +
                  np.trace(
                      sigma_real + sigma_generated - 2 * np.sqrt(np.outer(sqrt_eigval_real, sqrt_eigval_generated))))

    return fid


# Load the pre-trained Inception v3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Load and preprocess real and generated images
real_images = load_and_preprocess_real_images(num_images=10000)  # Load 10,000 real images
generated_image = load_and_preprocess_generated_image()  # Load one generated image

# Extract features from real and generated images
real_features = inception_model.predict(preprocess_images(real_images))
generated_features = inception_model.predict(preprocess_images(np.array([generated_image])))

# Calculate FID with one generated image and 10,000 real images
fid_score = calculate_fid(real_features, generated_features)
print(f"FID Score: {fid_score}")
