import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classification",
        description="Train wine classifier and optionally classify a single wine sample",
    )
    # Optional: classify a single wine with 13 numeric features
    parser.add_argument(
        "--wine",
        nargs=13,
        type=float,
        metavar=(
            "Alcohol",
            "Malic_acid",
            "Ash",
            "Alcalinity_of_ash",
            "Magnesium",
            "Total_phenols",
            "Flavanoids",
            "Nonflavanoid_phenols",
            "Proanthocyanins",
            "Color_intensity",
            "Hue",
            "OD280_OD315_of_diluted_wines",
            "Proline",
        ),
        help=(
            "Thirteen feature values of a single wine sample. "
            "If provided, Model 1 will classify this wine after training. "
            "Order: Alcohol Malic_acid Ash Alcalinity_of_ash Magnesium "
            "Total_phenols Flavanoids Nonflavanoid_phenols Proanthocyanins "
            "Color_intensity Hue OD280/OD315_of_diluted_wines Proline"
        ),
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    cols = [
        "Class",
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ]

    df = pd.read_csv("wine/wine.data", header=None, names=cols)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    # print(df_shuffled)
    wineClasses = df_shuffled["Class"].values
    hotOneWineClasses = tf.one_hot(wineClasses, 3)

    x = df_shuffled.drop(columns=["Class"])
    y = hotOneWineClasses
    X_train, X_test, y_train, y_test = train_test_split(
        x, y.numpy(), test_size=0.25, random_state=42
    )

    # MODEL 1
    epochs_m1 = 80
    learning_rate_m1 = 0.001
    batch_size_m1 = 32

    kernel_initializer_m1 = "random_normal"
    bias_initializer_m1 = "zeros"

    model_1 = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                8,
                activation="relu",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
            tf.keras.layers.Dense(
                16,
                activation="relu",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
            tf.keras.layers.Dense(
                13,
                activation="relu",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
            tf.keras.layers.Dense(
                8,
                activation="relu",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
            tf.keras.layers.Dense(
                5,
                activation="relu",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
            tf.keras.layers.Dense(
                3,
                activation="softmax",
                kernel_initializer=kernel_initializer_m1,
                bias_initializer=bias_initializer_m1,
            ),
        ]
    )

    model_1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate_m1)),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    history_m1 = model_1.fit(
        X_train,
        y_train,
        epochs=int(epochs_m1),
        batch_size=batch_size_m1,
        validation_split=0.25,
    )
    model_1.evaluate(X_test, y_test)

    # Model 2
    epochs_m2 = 80
    learning_rate_m2 = 0.001
    batch_size_m2 = 64

    kernel_initializer_m2 = "random_uniform"
    bias_initializer_m2 = "he_normal"

    model_2 = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                8,
                activation="relu",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
            tf.keras.layers.Dense(
                16,
                activation="relu",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
            tf.keras.layers.Dense(
                13,
                activation="relu",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
            tf.keras.layers.Dense(
                8,
                activation="relu",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
            tf.keras.layers.Dense(
                5,
                activation="relu",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
            tf.keras.layers.Dense(
                3,
                activation="softmax",
                kernel_initializer=kernel_initializer_m2,
                bias_initializer=bias_initializer_m2,
            ),
        ]
    )

    model_2.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=float(learning_rate_m2)),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    history_m2 = model_2.fit(
        X_train,
        y_train,
        epochs=int(epochs_m2),
        batch_size=batch_size_m2,
        validation_split=0.25,
    )
    model_2.evaluate(X_test, y_test)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_m1.history["loss"], label="Training Loss")
    plt.plot(history_m1.history["val_loss"], label="Validation Loss")
    plt.title("Model 1 Loss Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_m1.history["accuracy"], label="Training Accuracy")
    plt.plot(history_m1.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model 1 Accuracy Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_m2.history["loss"], label="Training Loss")
    plt.plot(history_m2.history["val_loss"], label="Validation Loss")
    plt.title("Model 2 Loss Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_m2.history["accuracy"], label="Training Accuracy")
    plt.plot(history_m2.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model 2 Accuracy Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    if args.wine is not None:
        feature_names = [
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]

        wine_features = np.array(args.wine, dtype=np.float32).reshape(1, 13)

        probs = model_1.predict(wine_features)
        predicted_class_index = int(np.argmax(probs[0]))
        predicted_class_label = predicted_class_index + 1  # dataset classes: 1, 2, 3

        print("Predicted wine class:", predicted_class_label)
        print("Class probabilities:", probs[0])
