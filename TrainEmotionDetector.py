import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def create_model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3),
               activation="relu", input_shape=(48, 48, 1), padding="same", strides=1),
        Conv2D(filters=64, kernel_size=(3, 3),
               activation="relu", padding="same", strides=1),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same",
               strides=1, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same",
               strides=1, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(7, activation="softmax")
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        learning_rate=0.0001), metrics=['accuracy'])

    return model


def train_model(model, train_generator, test_generator, epochs):
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator)
    )

    return model_info


def plot_accuracy(model_info):
    train_accuracy = model_info.history['accuracy']
    val_accuracy = model_info.history['val_accuracy']

    epochs = range(1, len(train_accuracy) + 1)
    plt.plot(epochs, train_accuracy, 'r', label='Train Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def main():
    train_data_generator = keras.preprocessing.image.ImageDataGenerator()
    test_data_generator = keras.preprocessing.image.ImageDataGenerator()

    train_generator = train_data_generator.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    test_generator = test_data_generator.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    model = create_model()

    model_info = train_model(model, train_generator,
                             test_generator, epochs=100)

    plot_accuracy(model_info)

    save_model(model)


main()
