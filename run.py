import cv2
import argparse
import knowledge_graph
from pprint import pprint
from modules.answer_generator import AnswerGenerator
from modules.paragraph_generator import ParagraphGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path of the input image', required=True)
args = vars(parser.parse_args())

image = cv2.imread(args['path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

knowledge, frame = knowledge_graph.create_knowledge_graph(image)
pprint(knowledge)

paragraph_generator = ParagraphGenerator()
paragraph = paragraph_generator.generate(knowledge)

answer_generator = AnswerGenerator()
print(paragraph)

while True:
    question = input().strip()
    if question == "":
        break
    print(answer_generator.predict(paragraph, question))
