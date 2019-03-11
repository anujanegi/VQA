from modules.object_detection import ObjectDetector
from modules.color_detection import ColorDetector
from collections import Counter, defaultdict
from pprint import pprint
import matplotlib.pyplot as plt
import cv2
import argparse

recursive_dict = lambda: defaultdict(recursive_dict)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path of the input image', required=True)
args = vars(parser.parse_args())

img = cv2.imread(args['path'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

"""
knowledge graph for the given image;
caption will be generated using this;
format:
{
    class1: {
        count: n1
        objects: {
            class1object1: {
                color: class1object1color
                location: [left, top, right, bottom] 
            },
            class1object2: {
                color: class1object2color
                location: [left, top, right, bottom]
            },
            ...
        }
    },
    ...
}
"""
knowledge = recursive_dict()

boxes, classes = ObjectDetector.predict(img)
class_count = Counter(classes)

# populate counts
for class_, count in class_count.items():
    knowledge[class_]["count"] = count

# populate object attributes
for i, box in enumerate(boxes):
    l, t, r, b = list(map(int, box))
    crop = img[t:b, l:r]
    color = ColorDetector.predict(crop)
    knowledge[classes[i]]['objects']["%s%d" % (classes[i], i)] = {
        "color": color,
        "location": [l, t, r, b]
    }

# plot
for i, box in enumerate(boxes):
    l, t, r, b = list(map(int, box))
    label = classes[i]
    # bounding boxes
    cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 3)
    # put labels
    cv2.putText(img, str(label), (l + 10, b - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

pprint(knowledge)
plt.imshow(img, 'gray')
plt.show()
