import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    X_train = np.load("dataset/X_train.npy")
    X_test = np.load("dataset/X_test.npy")
    y_train = np.load("dataset/y_train.npy")
    y_test = np.load("dataset/y_test.npy")

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(16, activation="relu", input_shape=(3,)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(2, activation="linear"))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001, momentum=0.5)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['accuracy'])
    model.summary()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=50, validation_split=0.1)

    for i in range(20):
        X = np.expand_dims(X_train[i], axis=0)
        output = model.predict(X)
        print(output, y_train[i])
