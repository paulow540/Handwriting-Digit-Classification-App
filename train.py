import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Show sample images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(training_images[i], cmap='gray')

plt.show()

# Build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])


# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(training_images, training_labels, epochs=10)

# Evaluate
loss, accuracy = model.evaluate(test_images, test_labels)

print(f"Accuracy: {accuracy}")

# Save model
model.save("handwritten_model.h5")

print("Model saved successfully.")