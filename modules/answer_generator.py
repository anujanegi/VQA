class AnswerGenerator:
    @staticmethod
    def predict(comprehension, question):
        """
        predict answer for a question based on the given data
        :param comprehension: data to read for answering the question
        :param question: question based on the comprehension
        :return: answer for the question
        """
        pass

ag = AnswerGenerator()
print(ag.predict("There are two people.", "How many people are there?"))
