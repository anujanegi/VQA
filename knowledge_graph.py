from modules.object_detection import ObjectDetector
from modules.color_detection import ColorDetector
from modules.text_detection import TextDetector
from modules.scene_classification import SceneClassifier
from collections import Counter
import cv2


def create_knowledge_graph(frame):
    """
    create a knowledge graph of the given image;
    will be used to generate a caption to pass to the qa system;
    format:
    {
        classes:{
            class1: {
                count: n1
                objects: {
                    class1object1: {
                        color: class1object1color,
                        location: [left, top, right, bottom],
                        text: ?
                    },
                    class1object2: {
                        color: class1object2color
                        location: [left, top, right, bottom],
                        text: ?
                    },
                    ...
                }
            },
            ...
        }
        scene:scene
    }
    :param frame: input image
    :return: knowledge dict and plotted image
    """

    knowledge = {"scene": "", "classes": {}}
    # add scene
    knowledge["scene"] = SceneClassifier.predict(frame)

    boxes, classes = ObjectDetector.predict(frame)
    class_count = Counter(classes)

    # populate counts
    for class_, count in class_count.items():
        knowledge["classes"][class_] = {}
        knowledge["classes"][class_]["count"] = count

    # populate object attributes
    for i, box in enumerate(boxes):
        l, t, r, b = list(map(int, box))
        crop = frame[t:b, l:r]
        text = TextDetector.detect(crop)
        color = ColorDetector.predict(crop)
        index = len(knowledge["classes"][classes[i]].get("objects", {}))
        if index == 0:
            knowledge["classes"][classes[i]]["objects"] = {}
        knowledge["classes"][classes[i]]["objects"]["%s%d" % (classes[i], index)] = {
            "color": color,
            "location": [l, t, r, b],
            "text": text
        }

    # plot
    for i, box in enumerate(boxes):
        l, t, r, b = list(map(int, box))
        label = classes[i]
        # bounding boxes
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 3)
        # put labels
        cv2.putText(frame, str(label), (l + 10, b - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return knowledge, frame
