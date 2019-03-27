import inflect
from num2words import num2words

class ParagraphGenerator:
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
        return "There %s %s %s. " % (
            self.ie.plural_verb("is", count=count),
            self.ie.number_to_words(count),
            self.ie.plural(class_, count=count)
        )

    def describe_object_color(self, classname, id, color):
        """
        describe the color of the object
        Example:
            The first person is wearing black.
            The first dog is brown in color.
            The second bottle is pink in color.
        :param classname: object to describe
        :param id: identifier of the object
        :param color: color of the object
        :return: sentence describing color
        """
        if(classname == "person"):
            template = "The %s %s is wearing %s color. "
        else:
            template = "The %s %s is %s in color. "

        return template % (
            num2words(id, to='ordinal'),
            classname,
            color
        )

        def describe_object_text(self, classname, id, text):
            """
            describe the text on the object
            Example:
                The first board says hello.
                Hello is written on the first stop sign.
            :param classname: object to describe
            :param id: identifier of the object
            :param text: text on the object
            :return: sentence describing text
            """
            description = ""

            description +=  "The %s %s says %s. " % (
                num2words(id, to='ordinal'),
                classname,
                text
            )
            description +=  "%s is written on the %s %s. " % (
                text,
                num2words(id, to='ordinal'),
                classname
            )

            return description

    def describe_scene(self, scene):
        """
        describe the scene
        Example:
            The picture is taken at a gas station.
            This is a gas station.
        :param scene: scene of the image
        :return: sentence describing count
        """
        description = ""

        description += "The picture is taken at %s. " % (
            self.ie.a(scene)
        )

        description += "This is %s. " % (
            self.ie.a(scene)
        )

        return description

    def describe_class(self, classname, attributes):
        class_description = ""
        class_description += self.describe_class_count(classname, attributes['count'])
        for i in range(attributes['count']):
            properties = attributes['objects']['%s%d' % (classname, i)]
            class_description += self.describe_object_color(classname, i+1, properties['color'])
            if(properties['text'] != ''):
                class_description += self.describe_object_text(classname, i+1, properties['text'])
        return class_description

    def generate(self, knowledge_graph):
        """
        generates a paragraph description from knowledge_graph
        :param knowledge_graph: knowledge graph of an image
        :return: generated paragraph description
        """
        paragraph_description = ""
        for class_ in knowledge_graph['classes']:
            class_description = self.describe_class(class_, knowledge_graph["classes"][class_])
            paragraph_description += class_description

        scene_description = self.describe_scene(knowledge_graph['scene'])
        paragraph_description += scene_description
        return paragraph_description
