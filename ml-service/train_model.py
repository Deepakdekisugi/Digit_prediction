"""Train and save a small CNN on MNIST."""
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(save_path='model/model.h5', epochs=5):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
    x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print('Saved model to', save_path)

if __name__ == '__main__':
    train()
