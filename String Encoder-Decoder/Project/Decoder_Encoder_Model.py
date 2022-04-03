import re
import struct
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import read

FS = 8 * 1000  # Sampling frequency.
DURATION = 40 / 1000  # Duration time = 1/Fs

FREQUENCIES = [100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 4000]
FREQUENCIES_FILTER = [400, 600, 800, 1000, 1200, 1600, 2000, 2400, 4000]

LOOK_UP_TABLE = {
    'A': [400, 800, 1600],
    'B': [400, 800, 2400],
    'C': [400, 800, 4000],
    'D': [400, 1200, 1600],
    'E': [400, 1200, 2400],
    'F': [400, 1200, 4000],
    'G': [400, 2000, 1600],
    'H': [400, 2000, 2400],
    'I': [400, 2000, 4000],
    'J': [600, 800, 1600],
    'K': [600, 800, 2400],
    'L': [600, 800, 4000],
    'M': [600, 1200, 1600],
    'N': [600, 1200, 2400],
    'O': [600, 1200, 4000],
    'P': [600, 2000, 1600],
    'Q': [600, 2000, 2400],
    'R': [600, 2000, 4000],
    'S': [1000, 800, 1600],
    'T': [1000, 800, 2400],
    'U': [1000, 800, 4000],
    'V': [1000, 1200, 1600],
    'W': [1000, 1200, 2400],
    'X': [1000, 1200, 4000],
    'Y': [1000, 2000, 1600],
    'Z': [1000, 2000, 2400],
    ' ': [1000, 2000, 4000]
}


def get_samples(letter, is_capital, amp=1):
    x_axis = np.linspace(0.0, DURATION, 320, endpoint=False)
    f0 = LOOK_UP_TABLE[letter][0]
    f1 = LOOK_UP_TABLE[letter][1]
    f2 = LOOK_UP_TABLE[letter][2]
    f3 = int(200 if is_capital is True else 100)
    y_axis = amp * np.cos(2 * np.pi * f0 * x_axis)
    y_axis += amp * np.cos(2 * np.pi * f1 * x_axis)
    y_axis += amp * np.cos(2 * np.pi * f2 * x_axis)
    y_axis += amp * np.cos(2 * np.pi * f3 * x_axis)
    return y_axis


def text_to_sound(text, fileName):
    line = re.sub(r'[^a-zA-Z ]', '', text)

    ls = np.array([])
    for letter in line:
        is_space = True if letter == ' ' else False
        is_capital = not letter.isspace() and letter.isupper()
        char = letter if is_space else letter.upper()
        ls = np.append(ls, get_samples(char, is_capital))

    with wave.open(fileName, "w") as file:
        file.setparams(
            (1, 2, 8000, len(ls), "NONE", "not compressed"))
        for i in ls:
            file.writeframes(struct.pack('h', int(i * 100)))


def sound_to_text_fft(sound_file):
    retrieved_text = ""
    fs, data = read(sound_file)
    number_of_samples = int(DURATION / (1 / FS))
    cnt = int(data.size) // number_of_samples
    for i in range(cnt):
        start = i * number_of_samples
        end = (i + 1) * number_of_samples
        current_duration = data[start: end]
        T = 1 / FS
        yf = fft(current_duration)
        xf = fftfreq(number_of_samples, T)[:number_of_samples // 2 + 1]
        y_in_frequency_domain = 2.0 / number_of_samples * np.abs(yf[0:number_of_samples // 2 + 1])
        vals = list(y_in_frequency_domain)
        frequencies = []
        temp = []
        for idx, val in enumerate(vals):
            temp.append((abs(xf[idx]), val))

        temp.sort(key=lambda tup: tup[1], reverse=True)
        for idx in range(0, 4):
            frequencies.append(temp[idx][0])

        frequencies = sorted(frequencies)

        for idx, freq in enumerate(frequencies):
            diff = 100000
            correct_freq = freq
            for fft_freq in FREQUENCIES:
                if abs(fft_freq - freq) < diff:
                    diff = abs(fft_freq - freq)
                    correct_freq = fft_freq
            frequencies[idx] = correct_freq
        is_found = False
        for key in LOOK_UP_TABLE:
            temp = []
            for val in LOOK_UP_TABLE[key]:
                temp.append(val)
            temp = sorted(temp)
            if key != ' ' and temp[0] == frequencies[1] and temp[1] == frequencies[2] and temp[2] == frequencies[3]:
                if frequencies[0] == 100:
                    retrieved_text += key.lower()
                else:
                    retrieved_text += key
                is_found = True
                break
            if key == ' ' and temp[0] == frequencies[1] and temp[1] == frequencies[2] and temp[2] == frequencies[3]:
                retrieved_text += key
                is_found = True
                break
        if not is_found:
            retrieved_text += '*'

    return retrieved_text


def sound_to_text_band_pass_filter(sound_file):
    retrieved_text = ""
    fs, data = read(sound_file)
    # data = data[::-1]

    number_of_samples = int(DURATION / (1 / FS))
    cnt = int(data.size) // number_of_samples

    for i in range(cnt):
        start = i * number_of_samples
        end = (i + 1) * number_of_samples
        current_duration = data[start: end]
        current_duration = np.divide(current_duration, np.amax(current_duration))
        ls = [0.0] * 320
        ar = np.array(ls)
        temp = []
        frequencies = []
        for freq in FREQUENCIES_FILTER:
            ar = band_pass_filter_iir(freq, current_duration, 5)
            temp.append((freq, max(ar)))

        capital = band_pass_filter_iir(200, current_duration, 5)
        small = band_pass_filter_iir(100, current_duration, 5)

        is_capital = False
        if max(capital) > max(small):
            is_capital = True

        temp.sort(key=lambda tup: tup[1], reverse=True)
        for i in range(0, 3):
            frequencies.append(temp[i][0])

        frequencies = sorted(frequencies)
        is_found = False
        for key in LOOK_UP_TABLE:
            temp = []
            for val in LOOK_UP_TABLE[key]:
                temp.append(val)
            temp = sorted(temp)
            if temp[0] == frequencies[0] and temp[1] == frequencies[1] and temp[2] == frequencies[2]:
                if not is_capital:
                    retrieved_text += key.lower()
                else:
                    retrieved_text += key
                is_found = True
                break
        if not is_found:
            retrieved_text += '*'

    return retrieved_text


# Plot the signal in frequency domain.
def plot_signal_in_freq_domain(y):
    T = 1 / (FS + 1000)
    yf = fft(y)
    t = int(DURATION * (FS + 1000))
    xf = fftfreq(t, T)[:t // 2]
    yf = 2.0 / t * np.abs(yf[0:t // 2])
    plt.plot(xf, yf)
    plt.grid()
    plt.show()


# Plot the signal in time domain.
def plot_signal_in_time_domain(y):
    x = np.linspace(0.0, DURATION, 320, endpoint=False)
    plt.plot(x, y)
    plt.grid()
    plt.show()


def band_pass_filter_iir(center, sig, Q):
    b, a = signal.iirpeak(center, Q, FS)
    y = signal.filtfilt(b, a, sig, axis=0)
    return y


if __name__ == "__main__":

    while True:
        print("\nList of options:")
        print("1) Encode a string")
        print("2) Decode a wave file")
        print("3) Exit\n")
        try:
            op = int(input("Choose an option: "))
            if op not in [1, 2, 3]:
                raise NameError("")
        except NameError as err:
            print("\nEnter a valid number\n")
            continue
        if op == 3:
            break
        if op == 1:
            st = input("Enter a string: ")
            outFileName = input("Enter name of the output file : ")
            if not outFileName.endswith(".wav"):
                outFileName += ".wav"
            text_to_sound(st, outFileName)
            print("The text was encoded successfully")
        elif op == 2:
            print("\nA) Using fourier transform")
            print("B) Using bandpass filters")
            print("C) Using both methods")
            print("D) Exit\n")
            decodeOp = input("Choose an option: ").upper()
            while decodeOp not in ["A", "B", "C", "D"]:
                decodeOp = input("Enter a valid option: ").upper()
            if decodeOp == 'D':
                continue
            inputFileName = input("Enter the name of the wave file: ")
            if not inputFileName.endswith(".wav"):
                inputFileName += ".wav"
            if decodeOp == "A":
                decodedText = sound_to_text_fft(inputFileName)
                print("\nThe string is: ", decodedText)
            elif decodeOp == "B":
                decodedText = sound_to_text_band_pass_filter(inputFileName)
                print("\nThe string is : ", decodedText)
            elif decodeOp == "C":
                FFTDecodedText = sound_to_text_fft(inputFileName)
                bandpassDecodedText = sound_to_text_band_pass_filter(inputFileName)
                print("\nThe string from fourier transform is : ", FFTDecodedText)
                print("The string from bandpass filters is  : ", bandpassDecodedText)
