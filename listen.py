import pyaudio
import numpy as np
import sys

from keras.engine.saving import load_model

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLERATE = 44100
BUFFER=4096

model = load_model('models/model.h5')

p = pyaudio.PyAudio()

print('listening')

stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLERATE,
                input=True, output=False,
                frames_per_buffer=BUFFER)

previous = np.zeros(SAMPLERATE / 2)

while(True):
    data = stream.read(SAMPLERATE / 2)
    data = np.fromstring(data, dtype=np.int16)
    current = np.concatenate((previous, data))
    previous = data


    current = (current - np.mean(current)) / np.std(current)
    current = current.reshape(1, SAMPLERATE, 1)
    predictions = model.predict(current)
    #sys.stdout.write(str(np.argmax(predictions)))
    print(str(np.argmax(predictions)) + ' (' + str(np.amax(predictions) * 100) + '%)')

quit()
