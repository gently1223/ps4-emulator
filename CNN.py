import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from  sklearn.metrics import precision_score
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

from fastai.imports import Path
from preprocess import GenreFeature
    
genre_features = GenreFeature()

# if all of the preprocessed files do not exist, regenerate them all for self-consistency
if (
    os.path.isfile(genre_features.X_preprocessed_data)
    and os.path.isfile(genre_features.y_preprocessed_data)
):
    print("Preprocessed files exist, deserializing npy files")
    genre_features.load_deserialize_data()
else:
    print("Preprocessing raw audio files")
    genre_features.load_preprocess_data()

print("Training X shape: " + str(genre_features.X.shape))
print("Training Y shape: " + str(genre_features.y.shape))

X = genre_features.X.reshape(-1, genre_features.X.shape[1], genre_features.X.shape[2], 1)
y = genre_features.y

test_size = 0.1
validation_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
 
input_shape = (X_train.shape[1], X_train.shape[2])

# Create network
print("Build CNN model ...")

batch_size = 32
num_epochs = 50
num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,33,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
history = model.fit(
    X_train,
    y_train,   
    validation_data=(X_validation, y_validation),
    batch_size=batch_size,
    epochs=num_epochs,
)

print(history.history.keys())
#  "Accuracy/loss"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model/validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model/validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','validation loss'], loc='upper left')
plt.show()

print("\nTesting ...")
score, accuracy = model.evaluate(
    X_test, y_test, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Creates a HDF5 file 'lstm_genre_classifier.h5'
model_filename = "./model/model_weights_cnn.h5"
print("\nSaving model: " + model_filename)
model.save(model_filename)
# Creates a json file
print("creating .json file....")
model_json = model.to_json()
f = Path("./model/model_cnn.json")
f.write_text(model_json)