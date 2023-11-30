import pyaudio
import numpy as np
import wave
import utils
import fifobuffer
import matplotlib.pyplot as plt
import network
import math


CHUNK = 512
DELAY_CHUNK = CHUNK * 2
FFT_SIZE = 1024*4
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000

NUM_SONGS = 9

p = pyaudio.PyAudio()

fft_buffer = fifobuffer.FifoBuffer(total_size=FFT_SIZE, chunk_size=CHUNK)
audio_delay_buffer = np.zeros(24 * CHUNK, dtype=np.float32)
audio_delay_buffer_index = 0

stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK
        )


window = np.hamming(FFT_SIZE)

# Get file name for song to play
def get_song_file(number):
    return 'song' + str(number) + '.wav'

def load_song(number):
    global wave_file
    wave_file = wave.open(get_song_file(number), 'rb')


def play_loaded_song():
    global audio_delay_buffer_index
    num_chunks = math.floor(wave_file.getnframes() / CHUNK) - 2
    for i in range(num_chunks):
        data = wave_file.readframes(CHUNK)
        data_array = np.frombuffer(data, dtype=np.int16)
        
        # Write data_array to the audio_delay_buffer starting at audio_delay_buffer_index
        audio_delay_buffer[audio_delay_buffer_index:audio_delay_buffer_index+DELAY_CHUNK] = data_array
        # Increment audio_delay_buffer_index by CHUNK
        audio_delay_buffer_index += DELAY_CHUNK
        # If audio_delay_buffer_index is greater than or equal to the length of audio_delay_buffer
        if audio_delay_buffer_index >= len(audio_delay_buffer):
            # Set audio_delay_buffer_index to 0
            audio_delay_buffer_index = 0
        
        # Get the audio data from the audio_delay_buffer
        f_data = audio_delay_buffer[audio_delay_buffer_index:audio_delay_buffer_index+DELAY_CHUNK]
        # Convert the audio data to bytes
        f_data = data.astype(np.int16).tobytes()
        
        
        stream.write(data)

        # Get the audio data but only the first channel
        audio_data = np.frombuffer(data, dtype=np.int16)[::2]

        # Convert the audio data to floating point numbers between -1.0 and 1.0
        audio_data = audio_data / 32768.0
        # Push the audio data to the FifoBuffer
        fft_buffer.push(audio_data)

        audio_data_windowed = fft_buffer.get_buffer() * window * 40.0

        # Remove DC offset
        audio_data_windowed = audio_data_windowed - np.mean(audio_data_windowed)

        fft_result = np.fft.fft(audio_data_windowed)
        fft_formatted = utils.format_fft_output(fft_result, sample_rate=RATE, min_freq=200, max_freq=20000, num_output_values=FFT_SIZE//2)
        #fft_formatted = utils.gate_freqs(fft_formatted, min_db=-30)
        #Blur the fft_formatted array using numpy
        fft_formatted = np.convolve(fft_formatted, np.ones(10)/10, mode='same')

        # Shrink the fft_formatted array using numpy down to 256 values
        fft_formatted = fft_formatted[::8]
        fft_formatted = fft_formatted.astype(np.float32)
        freq_scaler = np.zeros(256, dtype=np.float32)
        for i in range(256):
            p= i / 256.0
            freq_scaler[i] = 2 ** -p

        fft_formatted = fft_formatted * freq_scaler
        fft_formatted = fft_formatted.astype(np.float32)
        # Plot freqs
        # plt.clf()
        # plt.plot(fft_formatted)
        # plt.pause(0.001)

        #Normalize output to max of 0 and min of -100
        fft_formatted = np.clip(fft_formatted, -100, 0)

        network.sample_freqs = fft_formatted

song_id = 0
while True:
    load_song(song_id)
    play_loaded_song()
    song_id = (song_id + 1) % NUM_SONGS