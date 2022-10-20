import argparse
from activity_recognition import Args, main


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str, required=True, help="the logfile to parse")
    parser.add_argument("--logname", type=str, help="the name of the logfile")
    parser.add_argument("--plot", action="store_true", help="plot the results")
    parser.add_argument(
        "--window-size",
        type=int,
        help="the window size in seconds for the sliding window classification",
    )

    args = parser.parse_args()
    return Args(**vars(args))


if __name__ == "__main__":
    main(parse_args())
