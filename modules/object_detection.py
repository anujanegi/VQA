import cv2
import numpy as np
from utils.coco_label_reader import coco_label_reader
from utils.non_max_suppression import non_max_suppression

PATH_TO_MODEL_WEIGHTS = "./data/faster_rcnn_inception_v2_coco.pb"
PATH_TO_GRAPH = "./data/faster_rcnn_inception_v2_coco_graph.pbtxt"
PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt"


class ObjectDetector:

    @staticmethod
    def predict(frame):
        """
        predict objects in the frame
        :param frame: input image as numpy array
        :return: list of objects and bounding boxes [x0, y0, x1, y1]
        """
        cv_net = cv2.dnn.readNetFromTensorflow(PATH_TO_MODEL_WEIGHTS, PATH_TO_GRAPH)
        labels = coco_label_reader(PATH_TO_LABELS)

        rows, cols, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, size=(rows, cols), swapRB=True, crop=False)
        cv_net.setInput(blob)
        cv_out = cv_net.forward()
        boxes = []
        classes = []
        for detection in cv_out[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                class_ = int(detection[1])
                if left > right:
                    left, right = right, left
                if top > bottom:
                    top, bottom = bottom, top
                boxes.append([left, top, right, bottom])
                classes.append(labels[class_])
        return non_max_suppression(np.asarray(boxes), np.asarray(classes))
