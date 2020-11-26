import argparse
import struct
from PIL import Image
from serial import serial_for_url
import numpy as np


SERIAL_PARAMS = {
    "baudrate": 115200,
    "rtscts": True,
    "timeout": 10.0,
    "write_timeout": 10.0,
}

CATEGORIES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def classify(port, img):
    with Image.open(img) as image:
        pixels = np.asarray(image.getdata(), dtype=np.uint8).flatten()
    with serial_for_url(port, **SERIAL_PARAMS) as s:
        # Discard any previous output
        s.read_all()
        s.write(pixels.tobytes())
        response = struct.unpack("<" + "b" * 10, s.read(10))
    return response

def main():
    p = argparse.ArgumentParser()
    p.add_argument("port")
    p.add_argument("img")
    args = p.parse_args() 
    response = classify(args.port, args.img)
    for category, value in zip(CATEGORIES, response):
        print(f"{category}: {value}")


if __name__ == "__main__":
    main()
