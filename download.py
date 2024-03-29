import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data: Add a channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Save the preprocessed data as NumPy arrays
np.save("mnist_data/x_train.npy", x_train)
np.save("mnist_data/y_train.npy", y_train)
np.save("mnist_data/x_test.npy", x_test)
np.save("mnist_data/y_test.npy", y_test)

print("Dataset downloaded and saved successfully.")