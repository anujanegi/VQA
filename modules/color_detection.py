from sklearn.cluster import KMeans
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
        height, width, dim = frame.shape
        resized = np.resize(frame, (width * height, dim))
        # cluster
        k_means = KMeans(16)
        labels = k_means.fit_predict(resized)
        palette = k_means.cluster_centers_
        # create new image
        new_image = np.reshape(palette[labels], (width, height, palette.shape[1]))
        new_image = new_image.astype(np.uint8)
        # find dominant color
        unique, counts = np.unique(new_image.reshape(-1, new_image.shape[2]), return_counts=True, axis=0)
        sort_ix = np.argsort(counts)
        sort_ix = sort_ix[-1]
        
        return ColorDetector.get_colour_name(tuple(unique[sort_ix]))
