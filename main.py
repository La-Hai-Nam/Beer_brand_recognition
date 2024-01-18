##
# Program that creates an artificial neural net that recognizes beer brands
# Created: 16.11.2022
# Author: Leon Sobotta
#
# #

import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# IDE warning because of tensorflow.keras import are known bugs and can be ignored, code still works
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def main():
    ##
    # Main function, calls all other necessary functions.
    # createPlot(), dataAugmentation() and performanceBoost() are optional
    # when dataAugmentation() is used validation result is usually lower
    # performanceBoost() doesn't work perfectly
    #
    # #
    data = getData()
    trainData = trainDataset(data)
    validateData = validateDataset(data)
    classNames = getClassNames(trainData)
    # createPlot(trainData, classNames)
    getBatchShape(trainData)
    # performance = performanceBoost(trainData, validateData)
    model = trainModel(trainData, validateData)
    testNet(model, classNames)


def getData():
    ##
    # Gets the directory of the images
    #
    # @return directory of the images
    #
    # #
    data_dir = pathlib.Path("beer_ds/beer/")
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    return data_dir


# setting the batch size, img_height and img_width to fit the dataset
# the bigger the batch the more system memory is necessary
batch_size = 1
img_height = 150
img_width = 150


def trainDataset(data_dir):
    ##
    # Function to create the dataset for training the model
    # validation split of 0.2 -> 80% of the images for training and 20% for validation
    # seed is set to make results reproducible
    #
    # @param directory of the images
    #
    # @return training dataset
    #
    # #
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds


def validateDataset(data_dir):
    ##
    # Function to create the dataset for validation of the model
    # validation split of 0.2 -> 80% of the images for training and 20% for validation
    # seed is set to make results reproducible
    #
    # @param directory of the images
    #
    # @return validation dataset
    #
    # #
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return val_ds


def getClassNames(train_ds):
    ##
    # Function to get the class names of the dataset images (sub folder names of images)
    #
    # @param training dataset
    #
    # @return class names
    #
    # #
    class_names = train_ds.class_names
    print(class_names)
    return class_names


def createPlot(train_ds, class_names):
    ##
    # Optional function to create a plot with the first 9 images, one from each class
    #
    # @param training dataset, class names
    # #
    plt.figure(figsize=(10, 10))

    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


def getBatchShape(train_ds):
    ##
    # Optional function to get the shape of the image_batch and labels_batch tensor in this case:
    # image_batch: (1, 4032, 1816, 3)
    # which means this is a batch of 1 image of shape 2032x1816x3
    # last dimension refers to color channels (rgb)
    # labels_batch: (1,), these are corresponding labels to the 1 image
    #
    # @param training dataset
    #
    # #
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def performanceBoost(train_ds, val_ds):
    ##
    # Function to boost the performance of the model accessing the dataset
    # by using buffered prefetching the model can yield data from disk without having I/O become blocking
    # Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch.
    # This will ensure the dataset does not become a bottleneck while training the model.
    # Dataset.prefetch overlaps data preprocessing and model execution while training
    #
    # @param training dataset, validation dataset
    #
    # @return tuple containing optimized (training dataset, validation dataset)
    # #
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def trainModel(train_ds, val_ds):
    ##
    # Function that creates and trains the model and generates plots to evaluate the performance of the model
    #
    # @param training dataset, validation dataset
    #
    # @return model
    #
    # #
    num_classes = 8
    epoch = 15

    # Function to prevent overfitting, by creating augmented images from dataset if dataset is small
    data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                  input_shape=(img_height,
                                               img_width,
                                               3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ])

#    plt.figure(figsize=(10, 10))
#    for images, _ in train_ds.take(1):
#            for i in range(9):
#                augmented_images = data_augmentation(images)
#                ax = plt.subplot(3, 3, i + 1)
#              plt.imshow(augmented_images[0].numpy().astype("uint8"))
#               plt.axis("off")
#
    model = tf.keras.Sequential([
        data_augmentation, #optional
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch
    )

    model.summary()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epoch)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    tf.keras.models.save_model(model, 'model.pbtxt')
    converter = tf.lite.TFLiteConverter.from_keras_model(model = model)
    model_tflite = converter.convert()
    open("beerDetectionModel.tflite", "wb").write(model_tflite)

    return model


def testNet(model, class_names):
    ##
    # Function to test the model with a local sample image
    #
    # @param model, class names
    # #
    data_dir = "file:///home/leon/Documents/biererkennung/test_img/test_img_01.jpg"
    image_path = tf.keras.utils.get_file('test_img_01', origin=data_dir)

    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    resized_img = tf.image.resize(img, (150, 150))
    img_array = tf.keras.utils.img_to_array(resized_img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions)

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == "__main__":
    main()
