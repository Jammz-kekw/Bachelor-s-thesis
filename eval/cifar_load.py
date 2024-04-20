import numpy as np
import pickle
import json


def load_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def load_cifar10_dataset(folder_path):
    # Load training data
    train_data = load_cifar10(folder_path + '/data_batch_1')
    for i in range(2, 6):
        train_data[b'data'] = np.vstack((train_data[b'data'], load_cifar10(folder_path + f'/data_batch_{i}')[b'data']))
        train_data[b'labels'] += load_cifar10(folder_path + f'/data_batch_{i}')[b'labels']

    # Load test data
    test_data = load_cifar10(folder_path + '/test_batch')

    # Combine training and test data
    all_data = {
        'train': {
            'images': train_data[b'data'],
            'labels': train_data[b'labels']
        },
        'test': {
            'images': test_data[b'data'],
            'labels': test_data[b'labels']
        }
    }
    return all_data


# Usage
cifar10_data = load_cifar10_dataset('D:\Juraj\CIFAR10\cifar-10-batches-py')

for _, pair in enumerate(zip(cifar10_data['train']['images'], cifar10_data['train']['labels'])):
    image, label = pair

    print(len(image))
    print(label)
