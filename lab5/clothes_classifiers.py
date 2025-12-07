import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import os

def convert_to_negative(image_array: np.ndarray) -> np.ndarray:
    return 255 - image_array


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))  # Resize to match training data

    img_array = np.array(img).astype("float32")
    img_array = convert_to_negative(img_array)  # Convert to negative

    img_array /= 255.0

    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def validate_files(image_path: str) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists("dense_clothes_model.keras"):
        raise FileNotFoundError("Dense model file not found: dense_clothes_model.keras")
    if not os.path.exists("conv_clothes_model.keras"):
        raise FileNotFoundError("Convolutional model file not found: conv_clothes_model.keras")


def main():
    parser = argparse.ArgumentParser(description="Predict clothing class from image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    validate_files(args.image_path)

    dense_model = tf.keras.models.load_model("dense_clothes_model.keras")
    conv_model = tf.keras.models.load_model("conv_clothes_model.keras")

    processed_image = preprocess_image(args.image_path)

    dense_input = processed_image.squeeze(-1)
    conv_input = processed_image  # Conv model expects (1, 28, 28, 1)

    dense_prediction = dense_model.predict(dense_input)
    conv_prediction = conv_model.predict(conv_input)

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    dense_class = class_names[int(np.argmax(dense_prediction))]
    conv_class = class_names[int(np.argmax(conv_prediction))]

    print(f"Image path: {args.image_path}")
    print(f"Dense model prediction: {dense_class}")
    print(f"Convolutional model prediction: {conv_class}")


if __name__ == "__main__":
    main()
