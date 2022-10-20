import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
import os

from utils import parse_logfile

SENSOR_FREQUENCY = 100.0  # Hz
WINDOW_SIZE = 10  # s
MIN_AMPLITUDE = 250


def fourier_transform(data: np.ndarray, timestamps: np.ndarray):
    # calculate the fft of the data
    fft_data = fft.fft(data)
    # calculate the frequencies
    freq = fft.fftfreq(timestamps.shape[-1], d=1 / SENSOR_FREQUENCY)
    # sort the data and frequencies by frequency
    idx = np.argsort(freq)
    freq = freq[idx]
    fft_data = fft_data[idx]

    # remove all frequences below 0.05 Hz
    # idx = np.where(freq > 0.05)
    # freq = freq[idx]
    # fft_data = fft_data[idx]

    return freq, fft_data


def split_into_seconds(timestamps: np.ndarray, data: np.ndarray):
    # remove the last samples so that we have samples that are full seconds
    r = int(len(timestamps) % SENSOR_FREQUENCY)
    seconds_ts = np.split(timestamps[:-r], len(timestamps) // SENSOR_FREQUENCY)
    seconds_data = np.split(data[:-r], len(timestamps) // SENSOR_FREQUENCY)

    return np.array(seconds_ts), np.array(seconds_data)


def visualize_signals_and_fft(
    data: np.ndarray,
    timestamps: np.ndarray,
    freq: np.ndarray,
    fft_data: np.ndarray,
):

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(timestamps, data)
    axs[0].set_title("Accelerometer")
    axs[0].set_ylabel("Acceleration (m/s^2)")
    axs[1].plot(freq, np.abs(fft_data))
    axs[1].set_title("Accelerometer")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Frequency (Hz)")
    # set the xlim to 0.0 Hz to 5 Hz
    axs[1].set_xlim(0, 5)
    return fig, axs


def classify_activity(fft_data: np.ndarray, freq: np.ndarray):
    """Classify the activity based on the fft data.

    Returns
        str, the activity (standing still, walking, running)
    """

    if np.max(fft_data) < MIN_AMPLITUDE:
        return "standing still"

    # get the peaks
    peaks = fft_data[np.where(freq > 0.5)]
    # get the peak with the highest amplitude
    peak = np.max(peaks)
    # get the frequency of the peak
    peak_freq = freq[np.where(fft_data == peak)]
    # classify the activity based on the frequency
    if peak_freq < 2.0:
        return "walking"
    else:
        return "running"


def sliding_window_classification(
    data: np.ndarray, timestamps: np.ndarray, window_size: int = WINDOW_SIZE
):
    """Classify the activity based on a sliding window of the data.

    Args:
        data: the data to classify
        timestamps: the timestamps of the data
        window_size: the window size in seconds

    Returns:
        list, the activities
    """

    # split the data into seconds
    seconds_ts, seconds_data = split_into_seconds(timestamps, data)
    n_seconds = len(seconds_ts)
    for i in range(WINDOW_SIZE, n_seconds):
        # gather all the signal data
        start = i - WINDOW_SIZE
        timestamps = np.concatenate(seconds_ts[start : start + WINDOW_SIZE])
        data = np.concatenate(seconds_data[start : start + WINDOW_SIZE])
        # compute the fourier transform of the accelerometer data
        acc_freq, acc_fft = fourier_transform(data, timestamps)

        # classify the activity
        activity = classify_activity(acc_fft, acc_freq)
        print(f"\t Activity at {round(np.mean(timestamps))} s: {activity}")


def main():
    logfile_standing_still = "logs/sensorLog_20221020T074111.txt"
    logfile_walk_in_hand = "logs/sensorLog_20221020T074200.txt"
    logfile_walk_in_pocket = "logs/sensorLog_20221020T074253.txt"
    logfile_run_in_hand = "logs/sensorLog_20221020T074342.txt"
    logfile_run_in_pocket = "logs/sensorLog_20221020T074415.txt"

    logfiles = [
        logfile_standing_still,
        logfile_walk_in_hand,
        logfile_walk_in_pocket,
        logfile_run_in_hand,
        logfile_run_in_pocket,
    ]
    log_names = [
        "Standing still",
        "Walking in hand",
        "Walking in pocket",
        "Running in hand",
        "Running in pocket",
    ]

    for log_name, logfile in zip(log_names, logfiles):
        acc, acc_timestamps, _, _, _, _ = parse_logfile(logfile)

        # compute the rms value of the accelerometer
        acc_rms = np.linalg.norm(acc, axis=1)
        # remove the mean of the rms value
        acc_rms -= np.mean(acc_rms)

        # compute the fft of the rms value
        acc_freq, acc_fft = fourier_transform(acc_rms, acc_timestamps)
        # plot the rms value and the fft in each subplot
        # fig, axs = visualize_signals_and_fft(acc_rms, acc_timestamps, acc_freq, acc_fft)
        # fig.savefig(f"plots/{os.path.basename(logfile[:-4])}.png")

        print(f"Classifying {log_name}")
        print(f"Activity for entire activity: {classify_activity(acc_fft, acc_freq)}")

        # classify the activity based on a sliding window
        sliding_window_classification(acc_rms, acc_timestamps)


if __name__ == "__main__":
    main()
