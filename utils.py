import numpy as np
import matplotlib.pyplot as plt

def format_fft_output(fft_result, sample_rate=44100, min_freq=20, max_freq=20000, num_output_values=512):
    """
    Format the output of FFT, scale frequencies logarithmically in amplitude and frequency,
    and return a single array of 512 values using interpolation.

    Parameters:
    - fft_result: numpy array, output of FFT (1024 bins)
    - sample_rate: int, sample rate of the audio signal
    - min_freq: int, minimum frequency to consider in Hertz
    - max_freq: int, maximum frequency to consider in Hertz
    - num_output_values: int, number of values in the output array

    Returns:
    - formatted_output: numpy array, array of 512 values representing scaled frequencies and amplitudes
    """

    # Extract the relevant half of the FFT bins
    fft_result = fft_result[:len(fft_result)//2]

    # Calculate magnitude from real and imaginary parts
    magnitude = np.sqrt(np.real(fft_result)**2 + np.imag(fft_result)**2) / len(fft_result)

    # Blur the magnitude a bit
    # magnitude = np.convolve(magnitude, np.ones(10)/10, mode='same')

    # Apply logarithmic scaling for amplitude
    magnitude_db = 20 * np.log10(magnitude)

    # Generate linearly spaced frequencies
    freqs_linear = np.fft.fftfreq(len(fft_result), d=1/sample_rate)

    # Find the indices corresponding to the desired frequency range
    min_idx = np.argmax(freqs_linear >= min_freq)
    max_idx = np.argmax(freqs_linear >= max_freq)

    # Extract the relevant frequency range
    freqs_range = freqs_linear[min_idx:max_idx]

    # Apply logarithmic scaling for frequency
    scaled_freqs = 2 ** (np.log2(min_freq) + (np.log2(max_freq/min_freq) * np.linspace(0, 1, len(freqs_range))))

    # # Interpolate the scaled magnitudes to match the desired frequency range
    interpolated_magnitudes = np.interp(scaled_freqs, freqs_range, magnitude_db[min_idx:max_idx])

    # # Interpolate to get the final output array with 512 values
    formatted_output = np.interp(np.linspace(min_freq, max_freq, num_output_values), scaled_freqs, interpolated_magnitudes)

    return formatted_output

def generate_and_fft_signal(num_harmonics=3, num_samples=44100, sample_rate=44100):
    """
    Generate a signal with multiple harmonics, perform FFT, and plot the result.

    Parameters:
    - num_harmonics: int, number of harmonics in the signal
    - num_samples: int, number of samples in the signal
    - sample_rate: int, sample rate of the signal

    Returns:
    - fft_result: numpy array, FFT result of the generated signal
    """

    # Generate time values
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)

    # Generate a signal with multiple harmonics
    signal = np.sum([np.sin(2 * np.pi * (i + 1) * 440 * t) for i in range(num_harmonics)], axis=0)

    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)

    # Plot the signal and its FFT
    # plt.figure(figsize=(12, 6))

    # # Plot the signal
    # plt.subplot(2, 1, 1)
    # plt.plot(t, signal)
    # plt.title('Generated Signal with Harmonics')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')

    # # Plot the FFT result
    # plt.subplot(2, 1, 2)
    # freqs = np.fft.fftfreq(len(fft_result), d=1/sample_rate)
    # plt.plot(freqs, np.abs(fft_result))
    # plt.title('FFT of the Signal')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')

    # plt.tight_layout()
    # plt.show()

    return fft_result

def plot_scaled_data(scaled_data, min_freq=20, max_freq=20000):
    """
    Plot the scaled data using matplotlib.

    Parameters:
    - scaled_data: numpy array, array of scaled magnitudes in decibels
    - min_freq: int, minimum frequency to consider in Hertz
    - max_freq: int, maximum frequency to consider in Hertz
    """

    # Generate the corresponding frequencies
    #freqs = 2 ** (np.log2(min_freq) + (np.log2(max_freq/min_freq) * np.linspace(0, 1, len(scaled_data))))

    # Plot the scaled data
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scaled_data)), scaled_data)
    plt.title('Scaled Data vs. Frequency')
    plt.grid(True)
    plt.show()

def gate_freqs(data, min_db=-40):
    """
    Set all values below a certain threshold to negative infinity.

    Parameters:
    - data: numpy array, array of scaled magnitudes in decibels
    - min_db: int, minimum decibel value to consider

    Returns:
    - gated_data: numpy array, array of scaled magnitudes in decibels with values below the threshold set to zero
    """

    # Set all values below the threshold to zero
    gated_data = np.where(data < min_db, min_db, data)

    return gated_data

def apply_equal_loudness_curve(data):
    """
    Apply the equal loudness curve to the data.

    Parameters:
    - data: numpy array, array of scaled magnitudes in decibels

    Returns:
    - data: numpy array, array of scaled magnitudes in decibels with equal loudness curve applied
    """

    #Generate the fletcher-munson curve into a numpy array

if(__name__ == "__main__"):
    fft_raw = generate_and_fft_signal(num_harmonics=6, num_samples=2048*4, sample_rate=44100)
    fft_formatted = format_fft_output(fft_raw, sample_rate=44100, min_freq=20, max_freq=20000, num_output_values=1024*4)
    plot_scaled_data(fft_formatted, min_freq=20, max_freq=20000)


