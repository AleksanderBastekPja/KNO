import argparse

import tensorflow as tf
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    model = tf.keras.models.load_model("my_model.keras")

    parser = argparse.ArgumentParser(
        prog="main",
        description="Show the number from inputted image by user",
        epilog="bla bla",
    )
    parser.add_argument("-i", "--image")
    args = parser.parse_args()
    image_name = args.image

    image = tf.keras.utils.load_img(
        image_name, color_mode="grayscale", target_size=(28, 28)
    )
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    predictions = model.predict(input_arr)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    print(f"Przewidziana liczba: {predicted_digit} (pewność: {confidence:.2f})")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
