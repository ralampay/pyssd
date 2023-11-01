import sys
import argparse
import os
import datetime
import os.path
import json

from torch import mode

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from modules.detect_image import DetectImage
from modules.train import Train

mode_choices = [
    "train",
    "detect-image"
]

def main():
    parser = argparse.ArgumentParser(description="PySSD: Python implementation of SSD")

    parser.add_argument("--mode", help="Mode to be used", choices=mode_choices, type=str, required=True)
    parser.add_argument("--config-file", help="Config file", type=str)

    args = parser.parse_args()

    mode        = args.mode
    config_file = args.config_file

    print(f"Mode: {mode}")
    print(f"Config file: {config_file}")

    if mode == "train":
        with open(config_file) as json_file:
            params = json.load(json_file)

        cmd = Train(params=params)
        cmd.execute()
    elif mode == "detect-image":
        with open(config_file) as json_file:
            params = json.load(json_file)

        cmd = DetectImage(params=params)
        cmd.execute()
    else:
        raise ValueError("Invalid mode {}".format(mode))

if __name__ == '__main__':
    main()
