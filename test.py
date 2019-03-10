from object_detection import ObjectDetector
from text import text_generator
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("./images/image2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

od = ObjectDetector()
boxes, classes = od.predict(img)

for i, box in enumerate(boxes):
    l, t, r, b = list(map(int, box))
    label = classes[i]
    # bounding boxes
    cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 3)
    # put labels
    cv2.putText(img, str(label), (l + 10, b - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

print("\n".join(text_generator.describe_objects(classes)))
plt.imshow(img)
plt.show()
