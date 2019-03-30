import cv2
import numpy as np


class ColorDetector:

    colors = {"black": [0, 0, 0],
              "white": [255, 255, 255],
              "gray": [102, 102, 102],
              "maroon": [140, 0, 0],
              "red": [255, 0, 0],
              "orange": [255, 85, 0],
              "yellow": [255, 170, 0],
              "green": [0, 255, 0],
              "aqua": [0, 255, 242],
              "blue": [0, 85, 255],
              "purple": [102, 0, 255],
              "pink": [255, 0, 255]}

    @staticmethod
    def nearest_color(color):
        """
        find the nearest color
        :param color: (R, G, B)
        :return: nearest (R', G', B')
        """
        wrapper = np.uint8([[list(color)]])
        hsv_wrapper = cv2.cvtColor(wrapper, cv2.COLOR_RGB2HSV)
        hsv = hsv_wrapper[0, 0] + [10, 0, 0]
        h, s, v = hsv
        if v < 30:
            return ColorDetector.colors["black"]
        elif v < 80:
            return ColorDetector.colors["gray"]
        elif v > 190 and s < 27:
            return ColorDetector.colors["white"]
        elif s < 54 and v < 185:
            return ColorDetector.colors["gray"]
        elif h < 18:
            if v < 150:
                return ColorDetector.colors["maroon"]
            else:
                return ColorDetector.colors["red"]
        elif h < 26:
            return ColorDetector.colors["orange"]
        elif h < 34:
            return ColorDetector.colors["yellow"]
        elif h < 73:
            return ColorDetector.colors["green"]
        elif h < 102:
            return ColorDetector.colors["aqua"]
        elif h < 127:
            return ColorDetector.colors["blue"]
        elif h < 149:
            return ColorDetector.colors["purple"]
        elif h < 175:
            return ColorDetector.colors["pink"]
        else:
            return ColorDetector.colors["red"]

    @staticmethod
    def approximate_image(image):
        return np.asarray([[ColorDetector.nearest_color(pixel) for pixel in row] for row in image])

    @staticmethod
    def find_color(image):
        image = ColorDetector.approximate_image(image)
        (values, counts) = np.unique(image.reshape(-1, image.shape[2]), return_counts=True, axis=0)
        index = np.argmax(counts)
        color = values[index]
        for (key, value) in ColorDetector.colors.items():
            if (value == color).all():
                return key
        return "black"

