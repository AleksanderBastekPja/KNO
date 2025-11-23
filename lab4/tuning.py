import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt

def model_builder_adaptive_hypper_params(hp):
  model = tf.keras.Sequential()
  model.add(normalizer)
  # model.add(tf.keras.layers.Dense(13, activation='relu'))

  n_layers = hp.Int(
    "n_layers",
    min_value=1,
    max_value=13,
    step=1)

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  for i in range(1, n_layers):
      hp_units = hp.Int(f"units_{i}", min_value=3, max_value=200, step=2)
      model.add(tf.keras.layers.Dense(hp_units, activation='relu'))
  model.add(tf.keras.layers.Dense(3, activation='softmax'))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(hp_learning_rate)),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

  return model


def model_builder_static(learning_rate):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(
            8,
            activation='relu',
            kernel_initializer=kernel_initializer_m1,
            bias_initializer=bias_initializer_m1
        ),
        tf.keras.layers.Dense(8, activation='relu',
                              kernel_initializer=kernel_initializer_m1,
                              bias_initializer=bias_initializer_m1),
        tf.keras.layers.Dense(3, activation='softmax',
                              kernel_initializer=kernel_initializer_m1,
                              bias_initializer=bias_initializer_m1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    cols = [
        "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
        "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
        "OD280/OD315 of diluted wines", "Proline"
    ]

    df = pd.read_csv("wine/wine.data", header=None, names=cols)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    wineClasses = df_shuffled["Class"].values
    hotOneWineClasses = tf.one_hot(wineClasses, 3)

    x = df_shuffled.drop(columns=["Class"])
    y = hotOneWineClasses

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x.to_numpy().astype("float32"))
    X_train, X_test, y_train, y_test = train_test_split(x, y.numpy(), test_size=0.25, random_state=42)

    # MODEL 1
    epochs_m1 = 80
    learning_rate_m1 = 0.001
    batch_size_m1 = 8

    kernel_initializer_m1 = 'random_normal'
    bias_initializer_m1 = 'zeros'

    # model_1 = model_builder_static(learning_rate_m1)
    # history_m1 = model_1.fit(X_train, y_train, epochs=int(epochs_m1), batch_size=batch_size_m1)
    # model_1.evaluate(X_test, y_test)

    # MODEL 2

    tuner = kt.Hyperband(model_builder_adaptive_hypper_params,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    # make chempionship and gain the best performing model
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=80, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}, layer number {best_hps.get('n_layers')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model_m2 = tuner.hypermodel.build(best_hps)
    history = model_m2.fit(X_train, y_train, epochs=80, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

    eval_result = hypermodel.evaluate(X_train, y_train)
    print("[test loss, test accuracy]:", eval_result)

    # plt.plot(history_m1.history['accuracy'], label='Training Accuracy m1')
    #
    # if 'val_accuracy' in history_m1.history:
    #     plt.plot(history_m1.history['val_accuracy'], label='Validation Accuracy for m1')
    #
    # plt.title('Model Accuracy over Epochs for models')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.show()

