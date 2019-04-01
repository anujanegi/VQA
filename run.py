import cv2
import argparse
import knowledge_graph
from pprint import pprint
from modules.caption_generator import generate_caption
from modules.answer_generator import AnswerGenerator
from modules.paragraph_generator import ParagraphGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path of the input image', required=True)
args = vars(parser.parse_args())
caption = generate_caption(args['path'])
image = cv2.imread(args['path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

knowledge, frame = knowledge_graph.create_knowledge_graph(image)
pprint(knowledge)

paragraph_generator = ParagraphGenerator()
paragraph = paragraph_generator.generate(knowledge)


paragraph += caption

print(paragraph)
answer_generator = AnswerGenerator(verbose=True)

while True:
    print("[QUESTION] ", end="")
    question = input().strip()
    if question == "":
        break
    answer = answer_generator.predict(paragraph, question)[0]
    print("[ANSWER] %s" % answer)
