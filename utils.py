from dataclasses import dataclass
import numpy as np


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
