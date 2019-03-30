import cv2
import argparse
import knowledge_graph
from pprint import pprint
from multiprocessing.pool import ThreadPool
from modules.scene_captioning import scene_caption
from modules.answer_generator import AnswerGenerator
from modules.paragraph_generator import ParagraphGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path of the input image', required=True)
args = vars(parser.parse_args())

pool = ThreadPool(processes=1)
captioning_task = pool.apply_async(scene_caption, (args['path']))

image = cv2.imread(args['path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

knowledge, frame = knowledge_graph.create_knowledge_graph(image)
pprint(knowledge)

paragraph_generator = ParagraphGenerator()
paragraph = paragraph_generator.generate(knowledge)

paragraph += captioning_task.get()

print(paragraph)
answer_generator = AnswerGenerator(verbose=True)

while True:
    print("[QUESTION] ", end="")
    question = input().strip()
    if question == "":
        break
    answer = answer_generator.predict(paragraph, question)[0]
    print("[ANSWER] %s" % answer)
