import os
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


# Define the U-Net architecture using PyTorch
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        output = self.output_layer(x2)
        return output


# Custom dataset class to load and preprocess images
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None, limit=None, num_classes=4, grid_size=16):
        self.directory = directory
        self.transform = transform
        self.limit = limit
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.images, self.labels = self.load_images()

    def load_images(self):
        images = []
        labels = []
        loaded_count = 0

        for filename in os.listdir(self.directory):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image = read_image(os.path.join(self.directory, filename)).float()
                images.append(image)

                label = self.extract_label(filename)
                label_matrix = self.create_label_matrix(label)
                labels.append(label_matrix)

                loaded_count += 1
                if self.limit is not None and loaded_count >= self.limit:
                    break

        return images, labels

    def extract_label(self, file):
        return int(file[11]) if file[6:10] == 'test' else int(file[12])

    def create_label_matrix(self, label):
        label_matrix = np.zeros((self.num_classes, self.grid_size, self.grid_size))
        label_matrix[label, :, :] = 1
        return label_matrix

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label_matrix = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label_matrix


# Preprocess function for images
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize

    return image


# Process data using the custom dataset and transforms
def process_data(directory, batch_size, limit=None):
    dataset = CustomDataset(directory, limit=limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == '__main__':
    source_dir = "D:\FIIT\\Bachelor-s-thesis\\Dataset\\sliced\\IHC_train"
    test_dir = "D:\FIIT\\Bachelor-s-thesis\\Dataset\\sliced\\IHC_test"

    print("Num GPUs Available: ", torch.cuda.device_count())

    # Load and preprocess data
    batch_size = 16
    dataloader = process_data(source_dir, batch_size, limit=5000)

    # Define the model
    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # labels = labels.unsqueeze(0)

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), 'unet_model.pth')
