import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("handwritten_digit_model.h5")

# Load the test data from NumPy arrays
x_test = np.load("mnist_data/x_test.npy")
y_test = np.load("mnist_data/y_test.npy")

# Make predictions on the test images
predictions = model.predict(x_test)

# Get the predicted digit for each image
predicted_digits = np.argmax(predictions, axis=1)

# Display a few test images with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digits[i]}, Actual: {y_test[i]}")
    plt.axis('off')

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig("test_predictions.png")
plt.close()

# Calculate the overall test accuracy
test_accuracy = np.sum(predicted_digits == y_test) / len(y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")