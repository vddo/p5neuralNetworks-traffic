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