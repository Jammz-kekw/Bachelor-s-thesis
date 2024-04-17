import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cnn import CustomDataset, UNet


def evaluate_predictions(model, dataloader):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            predictions = model(images)
            all_predictions.append(predictions)
            all_labels.append(labels)
    return torch.cat(all_predictions), torch.cat(all_labels)


def metrics(predictions, labels):
    # Convert predictions and labels to numpy arrays
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Reshape predictions and labels to (N, C, H, W) where N is the number of samples
    predictions_np = predictions_np.reshape(-1, predictions_np.shape[-3], predictions_np.shape[-2],
                                            predictions_np.shape[-1])
    labels_np = labels_np.reshape(-1, labels_np.shape[-3], labels_np.shape[-2], labels_np.shape[-1])

    # Reshape to (N, H, W) and convert to class labels
    predictions_np = np.argmax(predictions_np, axis=1)
    labels_np = np.argmax(labels_np, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(labels_np.flatten(), predictions_np.flatten())
    print("Accuracy:", accuracy)

    # Calculate precision, recall, and F1-score
    precision = precision_score(labels_np.flatten(), predictions_np.flatten(), average='weighted')
    recall = recall_score(labels_np.flatten(), predictions_np.flatten(), average='weighted')
    f1 = f1_score(labels_np.flatten(), predictions_np.flatten(), average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == '__main__':
    test_dir = "D:\FIIT\\Bachelor-s-thesis\\Dataset\\sliced\\IHC_test"
    gan_gen_dir = "D:\FIIT\\Bachelor-s-thesis\\Dataset\\results_cut\\run_4x\\he_to_ihc"

    model = UNet()
    model.load_state_dict(torch.load('cnn_models/unet_model.pth'))
    model.eval()

    test_dataset = CustomDataset(test_dir, limit=2000)
    batch_size = 16
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions, labels = evaluate_predictions(model, test_dataloader)

    metrics(predictions, labels)

    gan_data = CustomDataset(gan_gen_dir, get_labels=False)
    gan_dataloader = DataLoader(gan_data, batch_size=batch_size, shuffle=False)

    model.eval()
    gan_predictions = []

    with torch.no_grad():
        for image in gan_dataloader:
            image, _ = image
            image = image.to(device)
            gan_prediction = model(image)
            gan_predictions.append(gan_prediction)

    gan_predictions = torch.cat(gan_predictions, dim=0)

    # print(gan_predictions)

    probs = torch.softmax(gan_predictions, dim=1)
    labels_gan = torch.argmax(probs, dim=1)

    labels_gan_float = labels_gan.float()
    image_labels = torch.mean(labels_gan_float, dim=(1, 2))

    print(image)