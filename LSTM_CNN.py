import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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

X =  genre_features.X
y =  genre_features.y

test_size = 0.1
validation_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
 
input_shape = (X_train.shape[1], X_train.shape[2])

print("Build LSTM RNN model ...")

model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.5, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.5, return_sequences=False))
model.add(Dense(units=y_train.shape[1], activation="softmax"))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 20  # num of training examples per minibatch
num_epochs = 150
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
model_filename = "./model/model_weights_LSTM.h5"
print("\nSaving model: " + model_filename)
model.save(model_filename)
# Creates a json file
print("creating .json file....")
model_json = model.to_json()
f = Path("./model/model_LSTM.json")
f.write_text(model_json)
