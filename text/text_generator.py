import inflect


class TextGenerator:
    def __init__(self):
        self.ie = inflect.engine()

    def describe_class_count(self, class_, count):
        """
        describe the class_ with respect to count
        Example:
            There is one person.
            There are two kites.
            There is one watch.
        :param class_: class of the object
        :param count: number of objects
        :return: sentence describing count
        """
        return "There %s %s %s." % (
            self.ie.plural_verb("is", count=count),
            self.ie.number_to_words(count),
            self.ie.plural(class_, count=count)
        )

    def describe_object_color(self, obj, count, color=None, position=None):
        """
        describe the color of the object
        Example:
            The dog is brown in color.
            The bottle is pink in color.
        :param obj: object to describe
        :param count: count of the objects
        :param position: relative position of the object if any
        :param color: color of the object
        :return: sentence describing color
        """
        pass


