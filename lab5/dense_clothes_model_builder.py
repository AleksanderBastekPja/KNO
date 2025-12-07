import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(f"Train images dimensions: {train_images.shape}")
    print(f"Test images dimensions: {test_images.shape}")

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(f"Train images count: {len(train_labels)}")

    dense_model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    dense_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = dense_model.fit(train_images, train_labels, epochs=5)

    dense_model.save("dense_clothes_model.keras")

    test_loss, test_acc = dense_model.evaluate(test_images, test_labels)
    print(f"Model Accuracy: {test_acc * 100}%")

    predictions = dense_model.predict(test_images)

    # Plot and save accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.title("Dense (Fully Connected) Neural Network - Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("dense_accuracy_plot.png")
    plt.close()

    # Plot and save loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.title("Dense (Fully Connected) Neural Network - Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("dense_loss_plot.png")
    plt.close()

    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(10, 10))
    disp.plot()
    plt.title("Dense (Fully Connected) Neural Network - Confusion Matrix")
    plt.savefig("dense_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
