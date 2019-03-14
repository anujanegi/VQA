from sklearn.cluster import KMeans
import numpy as np
import webcolors
from colorthief import ColorThief


class ColorDetector:

    @staticmethod
    def find_closest_color(requested_colour):
        """
        finds the closest color in the sixteen HTML4 color
        :param requested_colour: RGB value of color
        :return: RGB value of color closest among HTML4 colors
        """
        min_colours = {}
        for key, name in webcolors.html4_hex_to_names.items():
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
        dominant_color = ColorThief(frame)
        return color_thief.get_color(quality=1)
        # height, width, dim = frame.shape
        #
        # # get centre of frame
        # frame = frame[(height//4):(3*height//4), (width//4):(3*width//4), :]
        # height, width, dim = frame.shape
        #
        # frame_vector = np.reshape(frame, [height*width, dim])
        # k_means = KMeans(n_clusters=1)
        # k_means.fit(frame_vector)
        #
        # unique_l, counts_l = np.unique(k_means.labels_, return_counts=True)
        # sort_ix = np.argsort(counts_l)
        # sort_ix = sort_ix[::-1]
        # cluster_center = [int(i) for i in k_means.cluster_centers_[sort_ix][0]][::-1]
        #
        # return ColorDetector.get_colour_name(tuple(cluster_center))
