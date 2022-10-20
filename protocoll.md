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