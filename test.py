import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import os

# 1️⃣ Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# 2️⃣ Build a CNN model
def build_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# 3️⃣ Load existing model or train a new one
MODEL_PATH = "mnist_digit_recognizer.h5"

if os.path.exists(MODEL_PATH):
    print("📂 Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("🧠 Training new model...")
    model = build_model()
    model.fit(
        x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1
    )
    model.save(MODEL_PATH)
    print("💾 Model saved as mnist_digit_recognizer.h5")

# 4️⃣ Evaluate on test set (optional)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Test accuracy: {test_acc:.4f}")


# 5️⃣ Function to preprocess and predict custom image
def predict_digit(image_path):
    # Load image
    img = Image.open(image_path).convert("L")  # convert to grayscale
    img = img.resize((28, 28))  # resize to 28x28

    # Convert to numpy array
    img_array = np.array(img)

    # Invert colors if background is white
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize and reshape
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    print(f"🔢 Predicted Digit: {predicted_digit} (confidence: {confidence:.2f})")


# 6️⃣ Ask user for image file name
file_path = input("📸 Enter the path to your digit image file (e.g., digit.png): ")

if os.path.exists(file_path):
    predict_digit(file_path)
else:
    print("⚠️ File not found. Please check the path and try again.")
