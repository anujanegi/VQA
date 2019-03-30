import re
from subprocess import check_output

PATH_TO_CAPTION_SCRIPT = "../../DeepRNN/main.py"


def scene_caption(filename):
    """
    Caption an image frame.
    We are using an external python2 application to achieve this.
    :param filename: filename of the image
    :return: caption string
    """
    out = check_output(['python', PATH_TO_CAPTION_SCRIPT, '--test_image', filename])
    caption = re.findall(r'\*\*([A-Z a-z]+)\*\*', out)
    if len(caption) > 0:
        return caption[0]
    else:
        return ""
