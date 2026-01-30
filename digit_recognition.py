import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import random
index = random.randint(0, 9999)


print("Running latest version...")
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize data (0-255 → 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training started...")
model.fit(train_images, train_labels, epochs=5)

model.fit(train_images, train_labels, epochs=5)

model.save("digit_model.h5")
print("Model saved!")

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# Predict first test image
prediction = model.predict(test_images)

index = int(input("Enter an index from 0 to 9999 for prediction: "))
if index < 0 or index > 9999:
    print("Invalid index! Using index 0.")
    index = 0
    prediction = model.predict(test_images)

plt.clf()
# choose any number from 0–9999
plt.imshow(test_images[index], cmap='gray')
plt.title(f"Predicted Digit: {prediction[index].argmax()}")
plt.axis('off')
plt.show()




