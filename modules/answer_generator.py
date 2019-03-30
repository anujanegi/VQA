from deeppavlov import build_model, configs
from utils.nostderrout import nostderrout


class AnswerGenerator:
    def __init__(self, verbose=False):
        """
        create a model from pre-trained weights
        """
        if verbose:
            print("Loading model. This might take a while...")
        with nostderrout():
            self.model = build_model(configs.squad.squad)
        if verbose:
            print("Loaded model.")

    def predict(self, comprehension, question):
        """
        predict answer for a question based on the given data
        :param comprehension: data to read for answering the question
        :param question: question based on the comprehension
        :return: answer for the question
        """
        return self.model([comprehension], [question])[0]

