import numpy as np
import matplotlib.pyplot as plt

SENSOR_FREQUENCY = 100.0  # Hz


def parse_logfile(logfile: str):
    """
    Parse GYRO, MAGNETOMETER and ACCELEROMETER
    """
    with open(logfile, "r") as f:
        lines = f.readlines()

    acc = []
    acc_timestamps = []
    gyr = []
    gyr_timestamps = []
    mag = []
    mag_timestamps = []

    for line in lines:
        line = line.strip()
        if line == "":
            continue
        line = line.split("\t")
        if line[1] == "ACC":
            acc.append([float(x) for x in line[2:]])
            acc_timestamps.append(int(line[0]))
        elif line[1] == "GYR":
            gyr.append([float(x) for x in line[2:]])
            gyr_timestamps.append(int(line[0]))
        elif line[1] == "MAG":
            mag.append([float(x) for x in line[2:]])
            mag_timestamps.append(int(line[0]))

    # convert to numpy arrays
    acc = np.array(acc)
    gyr = np.array(gyr)
    mag = np.array(mag)
    acc_timestamps = np.array(acc_timestamps)
    gyr_timestamps = np.array(gyr_timestamps)
    mag_timestamps = np.array(mag_timestamps)

    # make the timestamps start from 0
    acc_timestamps -= acc_timestamps[0]
    gyr_timestamps -= gyr_timestamps[0]
    mag_timestamps -= mag_timestamps[0]

    # make time in seconds
    acc_timestamps = acc_timestamps / 1e3
    gyr_timestamps = gyr_timestamps / 1e3
    mag_timestamps = mag_timestamps / 1e3

    return acc, acc_timestamps, gyr, gyr_timestamps, mag, mag_timestamps


def split_into_seconds(timestamps: np.ndarray, data: np.ndarray):
    # remove the last samples so that we have samples that are full seconds
    r = int(len(timestamps) % SENSOR_FREQUENCY)
    seconds_ts = np.split(timestamps[:-r], len(timestamps) // SENSOR_FREQUENCY)
    seconds_data = np.split(data[:-r], len(timestamps) // SENSOR_FREQUENCY)

    return np.array(seconds_ts), np.array(seconds_data)


def visualize_fft(
    ax: plt.axes,
    freq: np.ndarray,
    fft_data: np.ndarray,
):

    ax.plot(freq, np.abs(fft_data))
    ax.set_title("Accelerometer")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")
    # set the xlim to 0.0 Hz to 5 Hz
    ax.set_xlim(0, 5)


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
    visualize_fft(axs[1], freq, fft_data)

    return fig, axs


def export_mp4(activites, output_path):

    activities = np.array(activities)
    activities[activities == "standing still"] = 0
    activities[activities == "walking"] = 1
    activities[activities == "running"] = 2
    activities = activities.astype(int)
    plt.plot(activities)
    plt.title(f"Activity classification of {args.logname}")
    plt.xlabel("Time [s]")
    plt.ylabel("Activity")
    plt.yticks([0, 1, 2], ["still", "walking", "running"])
    # Add extra space so that yticks are not cut off
    plt.subplots_adjust(left=0.15)
    plt.savefig(f"plots/{output_path}_.p")
    plt.show()

    pass
