import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("handwritten_digit_model.h5")

# Load new data or use the test dataset
# Here, we'll use the test dataset for demonstration
from tensorflow.keras.datasets import mnist
_, (x_test, y_test) = mnist.load_data()

# Preprocess the data: Normalize pixel values to [0, 1]
x_test = x_test / 255.0

# Reshape the data: Add a channel dimension
x_test = x_test.reshape(-1, 28, 28, 1)

# Make predictions on new images
predictions = model.predict(x_test)

# Get the predicted digit for each image
predicted_digits = np.argmax(predictions, axis=1)

# Display a few images with their predicted labels and save the plot
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digits[i]}, Actual: {y_test[i]}")
    plt.axis('off')

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig("predictions.png")
plt.close()