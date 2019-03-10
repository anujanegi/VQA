def coco_label_reader(FILEPATH):
    """
    parse a label map file and return a dictionary
    label map has entries like this:
    item {
        name: ""
        id: 0
        display_name: ""
    }
    :param FILEPATH: path to the label map
    :return: label dictionary
    """
    mapping = {}
    with open(FILEPATH, 'r') as file:
        data = file.readlines()
        for i in range(0, len(data), 5):
            id = int(data[i + 2].split(":")[1].strip()) - 1
            label = data[i + 3].split(":")[1].strip()[1:-1]
            mapping[id] = label
    return mapping
