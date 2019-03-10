from collections import Counter
import inflect


def describe_objects(objects):
    """
    describe that the scenery contains the given objects
    Example:
        There is a person.
        There are two boys.
        There is an elephant.
    :param objects: list of objects
    :return list of sentences
    """
    text = []
    objects = Counter(objects)
    ie = inflect.engine()
    for obj, count in objects.items():
        sentence = "There %s %s." % (
            ie.plural_verb("is", count=count),
            ie.a(ie.plural(obj, count=count), count=ie.number_to_words(count))
        )
        text.append(sentence)
    return text
