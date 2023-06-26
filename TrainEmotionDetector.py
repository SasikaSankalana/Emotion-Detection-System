import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam


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

# Try 1 - Training Accuracy - 0.32 Test Accuracy - 0.40
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3),
           input_shape=(48, 48, 1), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3),  activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3),  activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])


model.compile(loss='categorical_crossentropy', optimizer=Adam(
    learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

model_info = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator)//64,
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(test_generator)//64
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
