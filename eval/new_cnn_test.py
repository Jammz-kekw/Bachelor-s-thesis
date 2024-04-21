import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


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


# Step 2: Preprocess the Images
def preprocess_images(images):
    # Normalize pixel values to range [0, 1]
    images = images.astype(np.float32) / 255.0
    return images


# Step 3: Split the Dataset
data_dir = r"D:\FIIT\Bachelor-s-thesis\Dataset\sliced\IHC_train"
# data_dir = r"D:\FIIT\Bachelor-s-thesis\Dataset\ihc_merged"
images, labels = load_data(data_dir, max_images=10000)
images = preprocess_images(images)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# Step 4: Create Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128 * 16 * 16)  # Adjust input size based on image dimensions
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Step 5: Data Augmentation (Optional)
# Define data augmentation transforms if needed
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    # Add more transforms here (e.g., RandomHorizontalFlip, RandomRotation, etc.)
])

# Step 6: Create Data Loaders
train_dataset = CustomDataset(x_train, y_train, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


model = CNN(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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
print(f"Test Accuracy: {accuracy}")
