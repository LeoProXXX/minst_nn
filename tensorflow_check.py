import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy


def check_with_tensorflow():
    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    y_train = to_categorical(y_train)

    model = tf.keras.Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(10)
    ])

    model.compile(optimizer='SGD',
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)


if __name__ == '__main__':
    check_with_tensorflow()