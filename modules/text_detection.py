import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

PATH_FROZEN_GRAPH = "./data/frozen_east_text_detection.pb"


class TextDetector:

    @staticmethod
    def localize_post_process(prob, boxes):
        """
        process raw bounding boxes
        :param prob: probability of text
        :param boxes: raw boxes around text
        :return: boxes on the scaled image
        """
        rectangles = []
        confidences = []

        (num_rows, num_cols) = prob.shape[2:4]
        for y in range(0, num_rows):
            prob_data = prob[0, 0, y]
            coord = [boxes[0, i, y] for i in range(4)]
            angles = boxes[0, 4, y]
            for x in range(0, num_cols):
                if prob_data[x] < 0.5:  # confidence cutoff
                    continue

                h = coord[0][x] + coord[2][x]
                w = coord[1][x] + coord[3][x]

                # adjust to original params
                (offset_x, offset_y) = (x * 4.0, y * 4.0)
                angle = angles[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                end_x = int(offset_x + (cos * coord[1][x]) + (sin * coord[2][x]))
                end_y = int(offset_y - (sin * coord[1][x]) + (cos * coord[2][x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                rectangles.append((start_x, start_y, end_x, end_y))
                confidences.append(prob_data[x])

        return non_max_suppression(np.array(rectangles), probs=confidences)

    @staticmethod
    def localize_text(frame, scale=(320, 320)):
        """
        localize bounding boxes around text area
        :param frame: input image
        :param scale: resize factor
        :return: list of coordinates [x0, y0, x1, y1]
        """

        layers = [
            "feature_fusion/Conv_7/Sigmoid",  # output probability
            "feature_fusion/concat_3"]  # bounding box

        (height, width) = frame.shape[:2]
        ratio_height = height / scale[0]
        ratio_width = width / scale[1]
        resized = cv2.resize(frame, scale)
        blob = cv2.dnn.blobFromImage(resized, 1.0, scale, (0, 0, 0), swapRB=False, crop=False)
        model = cv2.dnn.readNet(PATH_FROZEN_GRAPH)
        model.setInput(blob)
        (prob, coord) = model.forward(layers)
        coord = TextDetector.localize_post_process(prob, coord)

        boxes = []
        for (start_x, start_y, end_x, end_y) in coord:
            start_x = max(0, int(start_x * ratio_width) - 10)
            start_y = max(0, int(start_y * ratio_height) - 10)
            end_x = int(end_x * ratio_width) + 10
            end_y = int(end_y * ratio_height) + 10
            boxes.append([start_x, start_y, end_x, end_y])
        return boxes

    @staticmethod
    def get_stitched_text(detected):
        """
        stitch the broken words into a sentence
        :param detected: array of [text, bounding box]
        :return: sentence of words
        """
        # TODO: complete this function
        sentence = ""
        for text, _ in detected:
            sentence += text.strip() + " "
        return sentence

    @staticmethod
    def detect(frame):
        """
        find text in the frame
        :param frame: input image
        :return: list of texts with coordinates [text, location]
        """
        detected = []
        boxes = TextDetector.localize_text(frame)
        for box in boxes:
            (start_x, start_y, end_x, end_y) = box
            crop = frame[start_y: end_y, start_x: end_x]
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            crop = cv2.GaussianBlur(crop, (5, 5), 0)
            _, crop = cv2.threshold(crop, 100, 150, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(crop, config='-l eng --oem 1 --psm 7')
            detected.append([text, box])
        stitched = TextDetector.get_stitched_text(detected)
        return stitched
