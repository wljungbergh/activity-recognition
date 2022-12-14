import os
from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

from utils import (
    SENSOR_FREQUENCY,
    export_mp4,
    parse_logfile,
    split_into_seconds,
    visualize_fft,
    visualize_signals_and_fft,
)

WINDOW_SIZE = 3  # s
MIN_AMPLITUDE = 80
RUNNING_AMPLITUDE = 700


@dataclass
class Args:
    logfile: str
    logname: Optional[str] = None
    window_size: int = WINDOW_SIZE
    min_amplitude: int = MIN_AMPLITUDE
    plot: bool = False
    export_mp4: bool = False


def fourier_transform(data: np.ndarray, timestamps: np.ndarray):
    # calculate the fft of the data
    fft_data = fft.fft(data)
    # calculate the frequencies
    freq = fft.fftfreq(timestamps.shape[-1], d=1 / SENSOR_FREQUENCY)
    # sort the data and frequencies by frequency
    idx = np.argsort(freq)
    freq = freq[idx]
    fft_data = fft_data[idx]

    return freq, np.abs(fft_data)


def classify_activity(
    fft_data: np.ndarray, freq: np.ndarray, plot: bool = False, fn: str = ""
) -> Literal["standing still", "walking", "running"]:
    """Classify the activity based on the fft data.

    Returns
        str, the activity (standing still, walking, running)
    """
    # remove all negative frequencies
    fft_data = fft_data[freq > 0.0]
    freq = freq[freq > 0.0]
    fft_data = fft_data[freq < 5.0]
    freq = freq[freq < 5.0]

    if np.max(abs(fft_data)) < MIN_AMPLITUDE:
        activity = "standing still"
    else:

        # get the peak with the highest amplitude
        peak_idx = np.argmax(fft_data)
        # get the frequency of the peak
        peak_freq = freq[peak_idx]
        # classify the activity based on the frequency
        if peak_freq < 0.4 and fft_data[peak_idx] < MIN_AMPLITUDE:
            activity = "standing still"
        elif peak_freq < 2.0 or fft_data[peak_idx] < RUNNING_AMPLITUDE:
            activity = "walking"
        else:
            activity = "running"

    if plot:
        assert fn != ""
        fig, ax = plt.subplots()
        visualize_fft(ax, freq, fft_data)
        fig.savefig(f"{fn}.png")
        plt.close(fig)

    return activity


def sliding_window_classification(
    data: np.ndarray,
    timestamps: np.ndarray,
    window_size: Optional[int] = WINDOW_SIZE,
    plot: bool = False,
    name: str = "",
):
    """Classify the activity based on a sliding window of the data.

    Args:
        data: the data to classify
        timestamps: the timestamps of the data
        window_size: the window size in seconds

    Returns:
        list, the activities
    """

    window_size = WINDOW_SIZE if window_size is None else window_size
    # split the data into seconds
    seconds_ts, seconds_data = split_into_seconds(timestamps, data)
    n_seconds = len(seconds_ts)
    activities = []
    for i in range(window_size, n_seconds):
        # gather all the signal data
        start = i - window_size
        timestamps = np.concatenate(seconds_ts[start : start + window_size])
        data = np.concatenate(seconds_data[start : start + window_size])
        # compute the fourier transform of the accelerometer data
        acc_freq, acc_fft = fourier_transform(data, timestamps)

        if plot:
            assert name != ""

        # classify the activity
        activity = classify_activity(
            acc_fft,
            acc_freq,
            plot=plot,
            fn=f"plots/{name}_step_{str(round(np.mean(timestamps))).zfill(3)}",
        )
        print(
            f"\t Activity at {str(round(np.mean(timestamps))).zfill(3)} s: {activity}"
        )
        activities.append(activity)
    return activities


def main(args: Args):
    if args.plot:
        os.makedirs("plots", exist_ok=True)
    if args.logname is None:
        args.logname = os.path.basename(args.logfile).split(".")[0]

    acc, acc_timestamps, _, _, _, _ = parse_logfile(args.logfile)

    # compute the rms value of the accelerometer
    acc_rms = np.linalg.norm(acc, axis=1)
    # remove the mean of the rms value
    acc_rms -= np.mean(acc_rms)

    print(f"Classifying {args.logname}")
    # classify the activity based on a sliding window
    activities = sliding_window_classification(
        acc_rms,
        acc_timestamps,
        window_size=args.window_size,
        plot=args.plot,
        name=args.logname,
    )

    if args.plot:
        # Plot activity classification over time
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
        plt.savefig(f"plots/{args.logname}_activity_classification.png")
        plt.show()

    if args.export_mp4:
        export_mp4(activities, "movie.mp4")


if __name__ == "__main__":
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

    # for log_name, logfile in zip(log_names, logfiles):
    #     args = Args(logfile, logname=log_name, plot=True)
    #     main(args)

    args = Args("logs/test-log.txt", logname="Test", plot=True, export_mp4=True)
    main(args)
