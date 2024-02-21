import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

fid_scores = []
interpretations = []

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features(image_path, model):
    image = load_and_preprocess_image(image_path)
    features = model.predict(image)
    return features.flatten()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    term1 = np.sum((mu1 - mu2)**2)
    outer_product = np.outer(mu1 - mu2, mu1 - mu2)
    term2 = np.trace(sigma1 + sigma2 - 2 * np.sqrt(np.outer(sigma1, sigma2)))
    return term1 + term2

def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
    return fid

def interpret_fid_score(fid_score):
    if fid_score < 0.05:
        return "Excellent"
    elif 0.05 <= fid_score < 0.1:
        return "Very Good"
    elif 0.1 <= fid_score < 0.2:
        return "Good"
    elif 0.2 <= fid_score < 0.3:
        return "Fair"
    else:
        return "Poor"

def select_random_images(directory, sample_size=50):
    all_files = os.listdir(directory)
    image_files = [file for file in all_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    if len(image_files) < sample_size:
        raise ValueError("Not enough images in the directory.")
    
    selected_images = random.sample(image_files, sample_size)
    return [os.path.join(directory, image) for image in selected_images]

# Load the pre-trained Inception v3 model
inception_model = InceptionV3(weights='imagenet', include_top=False)

# Specify the paths to the real and generated images directory
real_images_directory = "C:\\FIIT\\BP\\Dataset\\sliced\\HE_test"
generated_images_directory = "C:\\FIIT\\BP\\Dataset\\sliced\\HE_test"

# Get a random sample of images from each directory
real_sample = select_random_images(real_images_directory, sample_size=50)
generated_sample = select_random_images(generated_images_directory, sample_size=50)

# Iterate over the pairs of real and generated images, calculate FID, and store the results
for real_image_path, generated_image_path in zip(real_sample, generated_sample):
    real_features = extract_features(real_image_path, inception_model)
    generated_features = extract_features(generated_image_path, inception_model)
    
    fid_score = calculate_fid(real_features, generated_features)
    fid_scores.append(fid_score)
    
    interpretation = interpret_fid_score(fid_score)
    interpretations.append(interpretation)

# Convert interpretations to colors
interpretation_colors = {'Excellent': 'green', 'Very Good': 'blue', 'Good': 'yellow', 'Fair': 'orange', 'Poor': 'red'}
colors = [interpretation_colors[interpretation] for interpretation in interpretations]

# Plot the FID scores
plt.scatter(range(len(fid_scores)), fid_scores, c=colors)
plt.xlabel('Sample Index')
plt.ylabel('FID Score')
plt.title('FID Scores Visualization')

# Create custom legend with colored dots and interpretation
legend_labels = [f'{interpretation[0]}' for interpretation in interpretation_colors.items()]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in interpretation_colors.values()]
plt.legend(legend_handles, legend_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='center left')

plt.show()
