import cv2
import argparse
import knowledge_graph
import matplotlib.pyplot as plt
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path of the input image', required=True)
args = vars(parser.parse_args())

image = cv2.imread(args['path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

knowledge, frame = knowledge_graph.create_knowledge_graph(image)
pprint(knowledge)
plt.imshow(frame)
plt.show()
