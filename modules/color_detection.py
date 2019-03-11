from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import cv2
import webcolors

class ColorDetector:
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

    def get_colour_name(requested_colour):
        """
        get color name of passed RGB value
        :param requested_colour: RGB value of a color
        :return: named color of the RGB value
        """
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = find_closest_color(requested_colour)
            actual_name = None
        return closest_name

    def predict(frame):
        """
        predicts primary color in the frame
        :param frame: input image as numpy array
        :return: name of the color
        """
        height, width, dim = frame.shape

        # get centre of frame
        frame = frame[(height//4):(3*height//4), (width//4):(3*width//4), :]
        height, width, dim = frame.shape

        frame_vector = np.reshape(frame, [height*width, dim])
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(frame_vector)

        unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
        sort_ix = np.argsort(counts_l)
        sort_ix = sort_ix[::-1]
        cluster_center = [int(i) for i in kmeans.cluster_centers_[sort_ix][0]][::-1]

        return get_colour_name(tuple(cluster_center))
