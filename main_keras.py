import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    X_train = np.load("dataset/X_train.npy")
    X_test = np.load("dataset/X_test.npy")
    y_train = np.load("dataset/y_train.npy")
    y_test = np.load("dataset/y_test.npy")

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)))
    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation="relu"))
    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation="relu"))
    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation="relu"))
    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation="relu"))

    model.add(tf.keras.layers.Dense(128, activation="relu"))

    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))

    model.add(tf.keras.layers.Dense(128, activation="relu"))


    model.add(tf.keras.layers.Dense(64, activation="relu"))

    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(2, activation="linear"))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.000001, momentum=0.9)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['accuracy'])
    model.summary()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=200, shuffle=False)

    for i in range(20):
        X = np.expand_dims(X_train[i], axis=0)
        output = model.predict(X)
        print(output, y_train[i])

    model.evaluate(x=X_test, y=y_test, batch_size=32)
