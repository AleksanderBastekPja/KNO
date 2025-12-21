import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(latent_dim),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      tf.keras.layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

if __name__ == "__main__":
    images_dir = "images"

    dataset = tf.keras.utils.image_dataset_from_directory(
        images_dir,
        labels=None,
        image_size=(128, 128),
        batch_size=32
    )
    print(dataset)
    augmented_dataset = dataset.map(lambda x: (data_augmentation(x, training=True)), num_parallel_calls=tf.data.AUTOTUNE)

    images = np.concatenate([x for x in dataset], axis=0)
    augmented_images = np.concatenate([np.concatenate([x for x in augmented_dataset], axis=0) for _ in range(10)])

    images = np.concatenate([images, augmented_images], axis=0)

    images = images.astype("float32") / 255.0
    print(f"Total number of images after augmentation: {len(images)}")

    train_images, test_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    shape = test_images.shape[1:]
    print(shape)
    latent_dim = 64
    autoencoder = Autoencoder(latent_dim, shape)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    autoencoder.fit(train_images, train_images,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(test_images, test_images))

    encoded_imgs = autoencoder.encoder(test_images).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = len(test_images) // 4
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
