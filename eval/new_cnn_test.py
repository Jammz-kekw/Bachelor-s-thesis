import os
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


def sort_by_labels_and_normalize(images, labels):
    level_0 = []
    level_1 = []
    level_2 = []
    level_3 = []

    for image, label in zip(images, labels):
        image_sort = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        if label == 0:
            level_0.append(image_sort)
        elif label == 1:
            level_1.append(image_sort)
        elif label == 2:
            level_2.append(image_sort)
        elif label == 3:
            level_3.append(image_sort)

    mean_0 = get_mean(level_0)
    mean_1 = get_mean(level_1)
    mean_2 = get_mean(level_2)
    mean_3 = get_mean(level_3)

    normalized_images = []
    for image_rgb, label in zip(images, labels):
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(image_lab)

        current_l_mean = np.mean(l)

        if label == 0:
            ratio = (mean_0 - current_l_mean) / mean_0 * 100
        elif label == 1:
            ratio = (mean_1 - current_l_mean) / mean_1 * 100
        elif label == 2:
            ratio = (mean_2 - current_l_mean) / mean_2 * 100
        elif label == 3:
            ratio = (mean_3 - current_l_mean) / mean_3 * 100
        else:
            raise ValueError

        l_normalized = np.clip(l + ratio, 0, 100)

        lab_merged = cv2.merge([l_normalized, a, b])
        back_to_rgb = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
        normalized_images.append(back_to_rgb)

    return normalized_images


def get_mean(array):
    l_channels = []

    for image in array:
        L, A, B = cv2.split(image)
        l_channels.append(L)

    mean = np.mean(l_channels)

    return mean


def extract_label(file):
    return int(file[11]) if file[6:10] == 'test' else int(file[12])


# Step 1: Load the Images and Extract Labels
def load_data(data_dir, max_images=None):
    images = []
    labels = []
    num_loaded = 0

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(os.path.join(data_dir, filename))
            image = image.resize((256, 256))
            images.append(np.array(image))

            label = extract_label(filename)
            labels.append(label)

            num_loaded += 1
            if max_images is not None and num_loaded >= max_images:
                break

    return np.array(images), np.array(labels)


def preprocess_images(images):
    images = images.astype(np.float32) / 255.0
    return images


class CustomDataset(Dataset):
    def __init__(self, images, labels, flip_h=False, flip_v=False, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.flip_h = flip_h
        self.flip_v = flip_v

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.flip_h:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])

        if self.flip_v:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])

        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes=4, dropout_prob=0.5, l2_lambda=0.001):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    data_dir = r"D:\FIIT\Bachelor-s-thesis\Dataset\sliced\IHC_train"
    # data_dir = r"D:\FIIT\Bachelor-s-thesis\Dataset\ihc_merged"
    # data_dir = r"D:\FIIT\Bachelor-s-thesis\Dataset\norm_test"

    images, labels = load_data(data_dir, max_images=12000)
    images = preprocess_images(images)
    images = sort_by_labels_and_normalize(images, labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Step 5: Data Augmentation (Optional)
    # Define data augmentation transforms if needed
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])

    # Step 6: Create Data Loaders
    train_dataset = CustomDataset(x_train, y_train, transform=transform, flip_h=True, flip_v=True)
    test_dataset = CustomDataset(x_test, y_test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = CNN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=model.l2_lambda)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    torch.save(model.state_dict(), 'cnn_models/new_model_L2_flip.pth')
    print(f"Test Accuracy: {accuracy}")
