import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)



def create_parser():
    new_parser = argparse.ArgumentParser(
        prog="main",
        description="Show the number from inputted image by user",
        epilog="bla bla",
    )
    new_parser.add_argument("-e", "--epochs")
    new_parser.add_argument("-b", "--batch_size")
    new_parser.add_argument("-l", "--learning_rate")
    return new_parser


if __name__ == "__main__":
    DATA_PATH = "earthquake_data_tsunami.csv"

    cols = [
        "magnitude","cdi","mmi","sig","nst","dmin","gap","depth",
        "latitude","longitude","Year","Month","tsunami"
    ]

    df = pd.read_csv(DATA_PATH)
    df = df[cols].copy()

    df.head()

    x = df.drop(columns=["tsunami"])
    y = df["tsunami"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    parser = create_parser()
    command_args = parser.parse_args()
    epochs, batch_size, learning_rate = command_args.epochs, command_args.batch_size, command_args.learning_rate

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(x_train, y_train, epochs=int(epochs), batch_size=int(batch_size))
    model.evaluate(x_test, y_test)
