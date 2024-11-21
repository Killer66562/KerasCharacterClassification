from keras.api import Model, Sequential
from keras.api.layers import Input, Activation, Dropout, Dense, Conv2D, MaxPool2D, Flatten, Resizing, \
    Rescaling, RandomCrop, RandomFlip, RandomRotation, RandomTranslation, RandomBrightness
from keras.api.optimizers import SGD, Adam
from keras.api.utils import image_dataset_from_directory

from models import Crawler

import os
import asyncio
import random
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import time

IMG_HEIGHT = 512
IMG_WIDTH = 256

BATCH_SIZE = 50
EPOCHS = 100
LEARNING_RATE = 0.0006


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

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    output_tensor = Dense(labels_len, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)

    return model

async def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")

    crawler = Crawler()
    crawler.is_async = True
    crawler.keep_images_only = True

    crawler.api_key = "0c893db2cb9cacfb7ef5185a0d936ad047161c48dd3bcaffa6e5557e4807ef48"
    crawler.user_id = "1392188"
    crawler.page_min = 1
    crawler.page_max = 20
    crawler.page_size = 100
    crawler.max_tasks = 100
    crawler.wait_time_min = 2
    crawler.wait_time_max = 3

    with open("datasets/labels.txt", mode="r", encoding="utf8") as file:
        labels = file.readlines()
    labels_len = len(labels)

    datasets_root_path = os.path.abspath("datasets")

    datasets_raw_path = os.path.join(datasets_root_path, "raw")
    datasets_train_path = os.path.join(datasets_root_path, "train")
    datasets_validation_path = os.path.join(datasets_root_path, "validation")
    datasets_test_path = os.path.join(datasets_root_path, "test")
    
    for label in labels:
        crawler.tags_str = label
        crawler.download_folder_path = os.path.join(datasets_raw_path, label).strip()

        datasets_train_label_path = os.path.join(datasets_train_path, label).strip()
        datasets_validation_label_path = os.path.join(datasets_validation_path, label).strip()
        datasets_test_label_path = os.path.join(datasets_test_path, label).strip()

        if not os.path.exists(crawler.download_folder_path):
            urls = await crawler.get_image_urls_async()
            urls = random.sample(urls, k=500)
            await crawler.download_images_async(urls=urls)

        paths = os.listdir(crawler.download_folder_path)
        paths_len = len(paths)
        paths_train = random.sample(paths, k=int(paths_len * 0.6))
        paths_other = [path for path in paths if path not in paths_train]
        paths_other_len = len(paths_other)
        paths_validation = random.sample(paths_other, k=int(paths_other_len * 0.5))
        paths_test = [path for path in paths_other if path not in paths_validation]

        if not os.path.exists(datasets_train_label_path):
            os.mkdir(datasets_train_label_path)
        if not os.path.exists(datasets_validation_label_path):
            os.mkdir(datasets_validation_label_path)
        if not os.path.exists(datasets_test_label_path):
            os.mkdir(datasets_test_label_path)

        for path in paths_train:
            shutil.copyfile(os.path.join(crawler.download_folder_path, path).strip(), os.path.join(datasets_train_label_path, path).strip())
        for path in paths_validation:
            shutil.copyfile(os.path.join(crawler.download_folder_path, path).strip(), os.path.join(datasets_validation_label_path, path).strip())
        for path in paths_test:
            shutil.copyfile(os.path.join(crawler.download_folder_path, path).strip(), os.path.join(datasets_test_label_path, path).strip())

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

    data_augmentation = Sequential([
        RandomFlip(),
        RandomRotation(0.2),
        RandomBrightness(0.2)
    ])

    '''
    datasets_train = tf.data.Dataset.from_tensors(datasets_train)
    datasets_validation = tf.data.Dataset.from_tensors(datasets_validation)
    datasets_test = tf.data.Dataset.from_tensors(datasets_test)
    '''

    datasets_train = datasets_train.map(lambda x, y: (data_augmentation(x, training=True), y))

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