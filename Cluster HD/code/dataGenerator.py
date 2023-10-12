import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import time
import numpy as np

# Record the start time
start_time = time.time()

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Preprocessing the data...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

print("Creating ImageDataGenerator for data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("Loading pre-trained VGG-16 model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

print("Creating a new model with VGG-16 as the base...")
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

print("Freezing the weights of the VGG-16 base model...")
for layer in base_model.layers:
    layer.trainable = False

print("Compiling the model...")
model.compile(optimizer=Adam(lr=0.0001),  
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model with augmented data...")
datagen.fit(x_train)
model.summary()
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          steps_per_epoch=len(x_train) / 64,
          epochs=10,
          validation_data=(x_test, y_test))

print("Evaluating the model on the test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')

# Calculate the total time taken
end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')
from tensorflow import keras
feature_extractor = keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)

# Extract features from the training set
train_features = feature_extractor.predict(x_train)

# Extract features from the test set
test_features = feature_extractor.predict(x_test)
y_combined = np.concatenate((y_train, y_test))

# Combine train_features and test_features horizontally
features_combined = np.concatenate((train_features, test_features), axis=0)
print(features_combined.shape)
column_names = ['y'] + [f'feature_{i+1}' for i in range(features_combined.shape[1])]
print(len(column_names))
print(y_combined.shape)
df = pd.DataFrame(np.column_stack((np.argmax(y_combined,axis=1), features_combined)), columns=column_names)

df.to_csv('cfar10_256_vgg_layers.csv', index=False)
print("Train Features shape:", train_features.shape)
print("Test Features shape:", test_features.shape)
print("y_train  shape:", y_train.shape)
print("y_test  shape:", y_test.shape)