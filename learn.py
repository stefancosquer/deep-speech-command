import os
import random

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import MaxPool1D, Conv1D, Dropout, BatchNormalization, GlobalAvgPool1D
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.io.wavfile import read

SAMPLERATE = 44100
CLASSES = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model = Sequential()

model.add(Conv1D(64, 88, activation='relu', input_shape=(SAMPLERATE, 1)))
model.add(MaxPool1D(3))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv1D(64, 4, activation='relu'))
model.add(MaxPool1D(3))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv1D(128, 4, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(GlobalAvgPool1D())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

## Train

def generator(directory, batch_size, shuffle=False):
    rate, background = read('backgrounds/hospital.wav')
    files = [file for file in os.listdir(directory) if file.endswith(".wav")]
    np.random.shuffle(files)
    x = np.zeros((batch_size, SAMPLERATE, 1))
    y = np.zeros((batch_size, len(CLASSES)))
    while True:
        if (shuffle):
            np.random.shuffle(files)
        for i in range(batch_size):
            # choose random file
            rate, samples = read(directory + '/' + files[i])
            # pad randomly before and after
            before = 1 # random.randint(0, SAMPLERATE - samples.size)
            after = SAMPLERATE - samples.size - before
            samples = np.concatenate((np.zeros(before), samples, np.zeros(after)))
            # mix with background sounds
            idx = np.random.randint(0, len(background) - SAMPLERATE)
            samples = samples + background[idx:idx + SAMPLERATE]
            # normalize sound
            samples = (samples - np.mean(samples)) / np.std(samples)
            x[i] = samples.reshape(1, SAMPLERATE, 1)
            y[i] = to_categorical(files[i][0], len(CLASSES))
        yield x, y

history = model.fit_generator(
    generator('recordings', 9, shuffle=True),
    validation_data=generator('recordings', 9),
    steps_per_epoch=1,
    validation_steps=1,
    epochs=1000,
    verbose=1)


## Evaluate

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

## Test

def test(file):
    rate, samples = read('recordings/' + file)
    samples = (samples - np.mean(samples)) / np.std(samples)
    samples = np.concatenate((samples, np.zeros(SAMPLERATE - samples.size)))
    samples = samples.reshape(1, SAMPLERATE, 1)
    predictions = model.predict(samples)
    print(file + ' : ' + str(np.argmax(predictions)))


test('1_stefan_0.wav')
test('2_stefan_0.wav')
test('3_stefan_0.wav')
test('4_stefan_0.wav')
test('5_stefan_0.wav')
test('6_stefan_0.wav')
test('7_stefan_0.wav')
test('8_stefan_0.wav')
test('9_stefan_0.wav')

