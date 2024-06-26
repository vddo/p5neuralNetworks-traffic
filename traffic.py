import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.2


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    
    # tf.keras.utils.to_categorical() converts a vector of integers
    # to a binary class matrix. Categorising the labels
    labels = tf.keras.utils.to_categorical(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    # Standardize color values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Get a compiled neural network
    model = get_model()
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # Define list to store images and labels
    list_images = list()
    list_labels = list()
    
    # Loop over sub directories
    for subdir in os.listdir(data_dir):
        
        # Only take directories
        path_subdir = os.path.join(data_dir, subdir)
        if os.path.isdir(path_subdir):
            
            # Loop over files in subdirectory
            for file in os.listdir(path_subdir):
                
                # Only take files
                path_file = os.path.join(path_subdir, file)
                if os.path.isfile(path_file):
                    
                    # List images
                    
                    # Convert color from BGR as default in cv2 to RGB
                    list_images.append(
                        cv.cvtColor(
                            cv.resize(
                            cv.imread(
                                path_file
                            ), (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_LINEAR
                        ), cv.COLOR_BGR2RGB
                        )
                    )

                    # List labels
                    list_labels.append(int(subdir))
                    
    return((list_images, list_labels))
                    
                    

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Model version 08-01
    model = tf.keras.models.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(30, 30, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    main()
