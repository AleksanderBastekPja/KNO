from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Add a channel dimension for Conv2D: (N, 28, 28) -> (N, 28, 28, 1)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    conv_model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation="relu"),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    conv_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    conv_model.summary()

    history = conv_model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_split=0.1)
    conv_model.save('conv_clothes_model.keras')

    test_loss, test_acc = conv_model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    predictions = conv_model.predict(test_images)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Conv2D Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('conv2d_accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Conv2D Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('conv2d_loss_plot.png')
    plt.close()

    y_pred = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(10, 10))
    disp.plot()
    plt.title('Conv2D Model Confusion Matrix')
    plt.savefig('conv2d_confusion_matrix.png')
    plt.close()


if __name__ == "__main__":
    main()
