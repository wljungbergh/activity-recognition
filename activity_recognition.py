import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from utils import parse_logfile

SENSOR_FREQUENCY = 100.0  # Hz
WINDOW_SIZE = 10  # s


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


def visualize_signals(
    acc: np.ndarray,
    acc_timestamps: np.ndarray,
    gyr: np.ndarray,
    gyr_timestamps: np.ndarray,
    mag: np.ndarray,
    mag_timestamps: np.ndarray,
):

    # plot acc, gyr, mag in seperate subplots
    _, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].plot(acc_timestamps, acc)
    axs[0, 0].set_title("Accelerometer")
    axs[0, 0].set_ylabel("Acceleration (m/s^2)")
    axs[0, 0].legend(["x", "y", "z"])
    axs[1, 0].plot(gyr_timestamps, gyr)
    axs[1, 0].set_title("Gyroscope")
    axs[1, 0].set_ylabel("Angular velocity (rad/s)")
    axs[1, 0].legend(["x", "y", "z"])
    axs[2, 0].plot(mag_timestamps, mag)
    axs[2, 0].set_title("Magnetometer")
    axs[2, 0].set_ylabel("Magnetic field (uT)")
    axs[2, 0].legend(["x", "y", "z"])
    # plot the rms of the acc, gyr, mag in seperate subplots
    axs[0, 1].plot(acc_timestamps, np.linalg.norm(acc, axis=1))
    axs[0, 1].set_title("Accelerometer")
    axs[0, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].plot(gyr_timestamps, np.linalg.norm(gyr, axis=1))
    axs[1, 1].set_title("Gyroscope")
    axs[1, 1].set_ylabel("Angular velocity (rad/s)")
    axs[2, 1].plot(mag_timestamps, np.linalg.norm(mag, axis=1))
    axs[2, 1].set_title("Magnetometer")
    axs[2, 1].set_ylabel("Magnetic field (uT)")

    # create a new figure for the fft
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    # calculate the fft of the acc, gyr, mag
    acc_freq, acc_fft = fourier_transform(acc, acc_timestamps)
    gyr_freq, gyr_fft = fourier_transform(gyr, gyr_timestamps)
    mag_freq, mag_fft = fourier_transform(mag, mag_timestamps)
    # plot the fft of the acc, gyr, mag in seperate subplots
    axs[0, 0].plot(acc_freq, np.abs(acc_fft))
    axs[0, 0].set_title("Accelerometer")
    axs[0, 0].set_ylabel("Acceleration (m/s^2)")
    axs[0, 0].legend(["x", "y", "z"])
    axs[1, 0].plot(gyr_freq, np.abs(gyr_fft))
    axs[1, 0].set_title("Gyroscope")
    axs[1, 0].set_ylabel("Angular velocity (rad/s)")
    axs[1, 0].legend(["x", "y", "z"])
    axs[2, 0].plot(mag_freq, np.abs(mag_fft))
    axs[2, 0].set_title("Magnetometer")
    axs[2, 0].set_ylabel("Magnetic field (uT)")
    axs[2, 0].legend(["x", "y", "z"])
    # plot the fft of the rms of the acc, gyr, mag in seperate subplots

    acc_rms = np.linalg.norm(acc, axis=1)
    acc_rms -= np.mean(acc_rms)

    axs[0, 1].plot(
        acc_freq,
        np.abs(fourier_transform(acc_rms, acc_timestamps)[1]),
    )
    axs[0, 1].set_title("Accelerometer")
    axs[0, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].plot(
        gyr_freq,
        np.abs(fourier_transform(np.linalg.norm(gyr, axis=1), gyr_timestamps)[1]),
    )
    axs[1, 1].set_title("Gyroscope")
    axs[1, 1].set_ylabel("Angular velocity (rad/s)")
    axs[2, 1].plot(
        mag_freq,
        np.abs(fourier_transform(np.linalg.norm(mag, axis=1), mag_timestamps)[1]),
    )
    axs[2, 1].set_title("Magnetometer")
    axs[2, 1].set_ylabel("Magnetic field (uT)")

    plt.show()


def main():
    logfile_standing_still = "logs/sensorLog_20221020T074111.txt"
    logfile2_walk_in_hand = "logs/sensorLog_20221020T074200.txt"
    logfile3_walk_in_pocket = "logs/sensorLog_20221020T074253.txt"
    logfile4_run_in_hand = "logs/sensorLog_20221020T074342.txt"
    logfile5_run_in_pocket = "logs/sensorLog_20221020T074415.txt"

    acc, acc_timestamps, gyr, gyr_timestamps, mag, mag_timestamps = parse_logfile(
        logfile_standing_still
    )

    visualize_signals(acc, acc_timestamps, gyr, gyr_timestamps, mag, mag_timestamps)
    # # compute the rms of all signals
    # acc_rms = np.linalg.norm(acc, axis=1)
    # gyr_rms = np.linalg.norm(gyr, axis=1)
    # mag_rms = np.linalg.norm(mag, axis=1)

    # acc_freq, acc_fft = fourier_transform(acc, acc_timestamps)
    # # remove all negative frequencies
    # valid_acc_freq, valid_acc_fft = (
    #     acc_freq[: (len(acc_freq) // 2) + 1],
    #     acc_fft[: (len(acc_freq) // 2) + 1 :],
    # )
    # split into seconds
    exit()
    acc_timestamps_seconds, acc_seconds = split_into_seconds(acc_timestamps, acc)
    n_seconds = acc_seconds.shape[0]
    for i in range(WINDOW_SIZE, n_seconds):
        # gather all the signal data
        start = i - WINDOW_SIZE
        timestamps_ = np.concatenate(
            acc_timestamps_seconds[start : start + WINDOW_SIZE]
        )
        acc_ = np.concatenate(acc_seconds[start : start + WINDOW_SIZE])
        # get the rms
        acc_rms = np.linalg.norm(acc_, axis=1)
        # compute the fourier transform of the accelerometer data
        acc_freq, acc_fft = fourier_transform(acc_rms, timestamps_)
        # get the peaks

        # create a plot of the rms over time
        fig, axes = plt.subplots(2, 1, figsize=(10, 3))
        axes[0].plot(acc_freq, acc_fft)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[1].plot(timestamps_, acc_rms)
        axes[1].set_ylabel("RMS acceleration (m/s^2)")
        axes[1].set_xlabel("Time (s)")
        # show the plot in each iteration
        plt.show()


if __name__ == "__main__":
    main()
