import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dense


def load_images(directory, limit=None):
    images = []
    labels = []
    loaded_count = 0

    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = cv2.imread(os.path.join(directory, filename))
            images.append(image)

            label = extract_label(filename)
            labels.append(label)

            loaded_count += 1
            if limit is not None and loaded_count >= limit:
                break

    return np.array(images), np.array(labels)


def extract_label(file):
    return file[11] if file[6:10] == 'test' else file[12]


def unet(input_shape=(256, 256, 3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)

    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)

    pool = GlobalAveragePooling2D()(conv9)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(pool)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


def process_data(images, labels):
    images = images / 255.0

    num_classes = 4
    one_hot = tf.one_hot(labels, depth=num_classes)

    return images, one_hot


if __name__ == '__main__':
    source_dir = "D:\FIIT\\Bachelor-s-thesis\\Dataset\\sliced\\IHC_train"
    test_dir = "D:\FIIT\\Bachelor-s-thesis\\Dataset\\sliced\\IHC_test"

    images, labels = load_images(source_dir, limit=1000)
    images_test, labels_test = load_images(test_dir, limit=200)

    images, labels_one_hot = process_data(images, labels)
    images_test, labels_one_hot_test = process_data(images_test, labels_test)

    # Create and compile the model
    model = unet()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(images, labels_one_hot, batch_size=32, epochs=10, validation_split=0.2)

    test_loss, test_accuracy = model.evaluate(images_test, labels_one_hot_test)

    print(f'test loss - {test_loss}')
    print(f'test accuracy - {test_accuracy}')
