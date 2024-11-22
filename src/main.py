from keras.api import Model, Sequential
from keras.api.layers import Input, Activation, Dropout, Dense, Conv2D, MaxPool2D, Flatten, Resizing, \
    Rescaling, RandomCrop, RandomFlip, RandomRotation, RandomTranslation, RandomBrightness
from keras.api.optimizers import Adam
from keras.api.utils import image_dataset_from_directory

from models import Crawler

import os
import asyncio
import tensorflow as tf
import matplotlib.pyplot as plt
import time

IMG_HEIGHT = 512
IMG_WIDTH = 256

BATCH_SIZE = 50
EPOCHS = 100
LEARNING_RATE = 0.0005


def my_model(labels_len: int):
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(3, 3))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(3, 3))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(3, 3))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(3, 3))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(3, 3))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)

    x = Dense(4096, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    output_tensor = Dense(labels_len, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)

    return model

async def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")

    labels: list[str] = []
    with open("datasets/labels.txt", mode="r", encoding="utf8") as file:
        while True:
            line = file.readline()
            if line == "":
                break
            line = line.strip()
            if line == "":
                continue
            labels.append(line)

    labels_len = len(labels)

    datasets_root_path = os.path.abspath("datasets")

    datasets_train_path = os.path.join(datasets_root_path, "train")
    datasets_validation_path = os.path.join(datasets_root_path, "validation")
    datasets_test_path = os.path.join(datasets_root_path, "test")

    datasets_train = image_dataset_from_directory(
        datasets_train_path, 
        label_mode="categorical", 
        class_names=labels, 
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        interpolation="nearest", 
        batch_size=BATCH_SIZE
    )
    datasets_validation = image_dataset_from_directory(
        datasets_validation_path, 
        label_mode="categorical", 
        class_names=labels, 
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        interpolation="nearest", 
        batch_size=BATCH_SIZE
    )
    datasets_test = image_dataset_from_directory(
        datasets_test_path, 
        label_mode="categorical", 
        class_names=labels, 
        image_size=(IMG_HEIGHT, IMG_WIDTH), 
        interpolation="nearest", 
        batch_size=BATCH_SIZE
    )

    normalization = Sequential([
        Rescaling(1. / 255)
    ])

    data_augmentation = Sequential([
        RandomFlip(),
        RandomRotation(0.2),
        RandomBrightness(0.2)
    ])

    datasets_train = datasets_train.map(lambda x, y: (normalization(x), y))
    datasets_validation = datasets_validation.map(lambda x, y: (normalization(x), y))
    datasets_test = datasets_test.map(lambda x, y: (normalization(x), y))

    datasets_train = datasets_train.map(lambda x, y: (data_augmentation(x), y))

    model = my_model(labels_len)
    momemtum = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=momemtum, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    hist = model.fit(
        datasets_train, 
        epochs=EPOCHS, 
        validation_data=datasets_validation, 
    )

    result = model.evaluate(datasets_test)
    print("accuracy=%.4f" % result[1])

    current_time = int(time.time())
    model.save(f'model-{current_time}.keras')

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout(pad=4)
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())