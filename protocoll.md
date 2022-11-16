## Implementing load_data()
- function output = tuple (images, labels)
- images = list of all images data
- one image data set = np.ndarray with correct size
- labels = list of integers

### read images
- load one image as array
- image file names format: 0000x_000yy.ppm
- with x range from 0 to 4
- with yy range from 0 to 29



### modell_v01.02

model_dropout = Sequential([
    layers.Conv2D(16, 4, padding='same', activation='relu', input_shape=(40, 40, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

result:
- does not recognize snow 

### modell_v01.03 
changelog:
- image size from 40x40 to 50x50


### mode_v01.04

model_dropout = Sequential([
    layers.Conv2D(8, 4, padding='same', activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



### model_v01.05

model_dropout = Sequential([
    layers.Conv2D(8, 5, padding='same', activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


### model_v01.06

model_dropout = Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

### model_v01.07

model_dropout = Sequential([
    layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


epochs = 15
history = model_dropout.fit(
    x_train, y_train, 
    # validation_data=(x_test, y_test), 
    epochs=epochs
)

- looks like overfitting
- less epochs


### model_01.08 less epochs

model_dropout = Sequential([
    layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(30, 30, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


epochs = 10
history = model_dropout.fit(
    x_train, y_train, 
    # validation_data=(x_test, y_test), 
    epochs=epochs
)

Epoch 10/10
666/666 [==============================] - 3s 5ms/step - loss: 0.2149 - accuracy: 0.9311

- rec sign for freeze as lable 11 (give way sign) with 100 %


### model_02.01 smaller cnn

model_dropout = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 10
history = model_dropout.fit(
    x_train, y_train, 
    # validation_data=(x_test, y_test), 
    epochs=epochs
)

417/417 [==============================] - 1s 2ms/step - loss: 0.1170 - accuracy: 0.9741

- same result
- image res = 50 x 50
- rec sign for freeze as lable 20 (turn right) with 100 %


### model_02.02 more filter

model_1 = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 9
history = model_1.fit(
    x_train, y_train, 
    # validation_data=(x_test, y_test), 
    epochs=epochs
)

417/417 [==============================] - 2s 5ms/step - loss: 0.1612 - accuracy: 0.9635


### model_02.03 filter size = 2x2

model_1 = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 9
history = model_1.fit(
    x_train, y_train, 
    validation_data=(x_test, y_test), 
    epochs=epochs
)

loss: 0.0762 - accuracy: 0.9809 - val_loss: 0.2250 - val_accuracy: 0.9409

- 18 epochs resulted in a better result
- raise epoch at least to 10



### model_03.01 second conv2d layer

test_size=0.3

model_1 = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 10

- 12 epochs resulted in better result
- try one more con2d layer



