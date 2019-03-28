from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import webcolors

class ColorDetector:

    @staticmethod
    def find_closest_color(requested_colour):
        """
        finds the closest color in the sixteen HTML4 color
        :param requested_colour: RGB value of color
        :return: RGB value of color closest among HTML4 colors
        """
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    @staticmethod
    def get_colour_name(requested_colour):
        """
        get color name of passed RGB value
        :param requested_colour: RGB value of a color
        :return: named color of the RGB value
        """
        try:
            closest_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = ColorDetector.find_closest_color(requested_colour)
        return closest_name

    @staticmethod
    def predict(frame):
        """
        predicts primary color in the frame
        :param frame: input image as numpy array
        :return: name of the color
        """

        frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
        #cluster and assign labels to the pixels
        clt = KMeans(n_clusters = 4)
        labels = clt.fit_predict(frame)
        #count labels to find most popular
        label_counts = Counter(labels)
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

        return ColorDetector.get_colour_name(tuple(np.array(dominant_color, dtype=int)))
