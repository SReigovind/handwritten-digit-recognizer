import numpy as np
import tensorflow as tf

# Load the preprocessed data from NumPy arrays
x_train = np.load("mnist_data/x_train.npy")
y_train = np.load("mnist_data/y_train.npy")
x_test = np.load("mnist_data/x_test.npy")
y_test = np.load("mnist_data/y_test.npy")

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on the testing dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model in the native Keras format with .h5 extension
model.save("handwritten_digit_model.h5")