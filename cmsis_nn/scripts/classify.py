import argparse
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def show_classify_result(img, response):
    fig, (img_ax, bar_ax) = plt.subplots(2, 1)
    image = mpimg.imread(img)
    img_ax.imshow(image)
    img_ax.set_axis_off()
    y_pos = np.arange(len(CATEGORIES))
    bar_ax.barh(y_pos, response, align="center")
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(CATEGORIES)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("Score")
    plt.show()


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
    p.add_argument("--gui", action="store_true")
    args = p.parse_args() 
    response = classify(args.port, args.img)
    if args.gui:
        show_classify_result(args.img, response)
    else:
        for category, value in zip(CATEGORIES, response):
            print(f"{category}: {value}")


if __name__ == "__main__":
    main()
