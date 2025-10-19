import argparse

import tensorflow as tf
import numpy as np


def create_parser():
    new_parser = argparse.ArgumentParser(
        prog="main",
        description="Show the number from inputted image by user",
        epilog="bla bla",
    )
    new_parser.add_argument("-i", "--image")
    return new_parser


if __name__ == "__main__":
    model = tf.keras.models.load_model("my_model.keras")

    parser = create_parser()
    command_args = parser.parse_args()
    image_name = command_args.image

    image = tf.keras.utils.load_img(
        image_name, color_mode="grayscale", target_size=(28, 28)
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_np_array = np.array([image_array])

    predictions = model.predict(image_np_array)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    print(f"Przewidziana liczba: {predicted_digit} (pewność: {confidence:.2f})")
