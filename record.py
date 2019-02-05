import pyaudio
import math
import time
import wave
import os
from scipy.io.wavfile import read, write
from collections import defaultdict

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLERATE = 44100
BUFFER = 1024
DELAY_BETWEEN_NUMBERS = 3
REPEATS_PER_NUMBER = 4

p = pyaudio.PyAudio()

name = raw_input("Votre nom: ")

recording = 'speech/recordings/speech_' + name + '.wav'
wavefile = wave.open(recording, 'wb')
wavefile.setnchannels(CHANNELS)
wavefile.setsampwidth(pyaudio.get_sample_size(FORMAT))
wavefile.setframerate(SAMPLERATE)


def record(in_data, frame_count, time_info, status_flags):
    wavefile.writeframes(in_data)
    return (None, pyaudio.paContinue)


def record_numbers():
    nums = [str(i) for i in range(1, 10) for set_num in range(REPEATS_PER_NUMBER)]
    for i in range(len(nums)):
        target = int(round(math.pi * i)) % len(nums)
        (nums[i], nums[target]) = (nums[target], nums[i])

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLERATE,
                    input=True, output=False,
                    frames_per_buffer=BUFFER,
                    stream_callback=record)

    print("Pret ?")
    time.sleep(DELAY_BETWEEN_NUMBERS)

    for num in nums:
        print(num)
        time.sleep(DELAY_BETWEEN_NUMBERS)

    stream.stop_stream()
    stream.close()
    p.terminate()
    wavefile.close()

    return nums


def trim(data):
    start = 0
    end = len(data) - 1

    mag = abs(data)
    thresold = mag.max() * 0.2

    for idx, point in enumerate(mag):
        if point > thresold:
            start = max(start, idx - 4410)
            break

    for idx, point in enumerate(mag[::-1]):
        if point > thresold:
            end = min(end, len(data) - idx + 4410)
            break

    return data[start:end]


def split(numbers):

    rate, data = read(recording)

    counts = defaultdict(lambda: 0)

    for i, label in enumerate(numbers):
        label = str(label)
        start_idx = (i + 1) * int(SAMPLERATE * DELAY_BETWEEN_NUMBERS)
        stop_idx = start_idx + int(SAMPLERATE * DELAY_BETWEEN_NUMBERS)

        digit = data[start_idx:stop_idx]
        digit = trim(digit)

        write('speech/recordings' + os.sep + label + "_" + name + "_" + str(counts[label]) + '.wav', SAMPLERATE, digit)

        counts[label] += 1

    os.remove(recording)


split(record_numbers())

