import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from new_cnn_test import CustomDataset, CNN, load_data, preprocess_images, sort_by_labels_and_normalize


def get_predictions(model, dataloader, device):
    """
        Used to get the predictions from trained model


    """

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            all_predictions.append(predictions)
            all_labels.append(labels)
    return torch.cat(all_predictions), torch.cat(all_labels)


def metrics(predictions, labels):
    """
        Used to calculate the statistic metrics

    """

    labels_np = labels.detach().cpu().numpy()
    predicted_classes = torch.argmax(predictions, dim=1)
    predicted_classes_np = predicted_classes.detach().cpu().numpy()

    accuracy = accuracy_score(labels_np, predicted_classes_np)
    precision = precision_score(labels_np, predicted_classes_np, average='weighted', zero_division=0)
    recall = recall_score(labels_np, predicted_classes_np, average='weighted', zero_division=0)
    f1 = f1_score(labels_np, predicted_classes_np, average='weighted', zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == '__main__':
    test_dir = r"D:\FIIT\\Bachelor-s-thesis\\Dataset\\sliced\\IHC_test"
    gan_gen_dir = r"D:\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\he_to_ihc"

    model = CNN()
    model.load_state_dict(torch.load('cnn_models/best_model.pth'))
    model.eval()

    images, labels = load_data(test_dir, max_images=100)
    images = preprocess_images(images)
    images = sort_by_labels_and_normalize(images, labels)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataset(images, labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions, labels = get_predictions(model, test_loader, device)

    metrics(predictions, labels)

    images_gan, labels_gan = load_data(gan_gen_dir)
    images_gan = preprocess_images(images_gan)

    gan_dataset = CustomDataset(images=images_gan, transform=transform)
    gan_dataloader = DataLoader(gan_dataset, batch_size=32, shuffle=False)
    gan_predictions_model, gan_labels = get_predictions(model, gan_dataloader, device)

    gan_predicted_labels = torch.argmax(gan_predictions_model, dim=1)
    gan_predicted_labels_np = gan_predicted_labels.cpu().numpy()

    plt.imshow(images_gan[17])
    plt.axis('off')
    plt.title(f'Predikovaná HER2 úroveň: {gan_predicted_labels_np[17]}')
    plt.show()
