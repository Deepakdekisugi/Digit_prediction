"""Train and save an improved CNN on MNIST with enhanced augmentation."""
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def build_model():
    """Build improved CNN model for digit recognition with better architecture"""
    
    model = Sequential([
        # First conv block - More filters for better feature extraction
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        
        # Second conv block - Deeper features
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        
        # Third conv block - Even more complex features
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(save_path='model/model.h5', epochs=20):
    """Train model with data augmentation for better generalization"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
    x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Data augmentation for better generalization
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

    model = build_model()
    print("Training model with data augmentation...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc*100:.2f}%')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print('Saved model to', save_path)

if __name__ == '__main__':
    train()
