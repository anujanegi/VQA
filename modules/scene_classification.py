import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

PATH_TO_MODEL_WEIGHTS = "./data/resnet18_places365.pth.tar"
PATH_TO_LABELS = "./data/categories_places365.txt"

class SceneClassifier:

    @staticmethod
    def load_files():
        """
        loads pre-trained weights, image transformer and class labels
        :return: model, image transformer and classes
        """
        # load pre-trained weights
        model = models.__dict__['resnet18'](num_classes=365)
        checkpoint = torch.load(PATH_TO_MODEL_WEIGHTS, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        # load image transformer
        image_transformer = trn.Compose([
                    trn.Resize((256,256)),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # load class labels
        classes = list()
        with open(PATH_TO_LABELS) as labels_file:
            for line in labels_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
        return model, image_transformer, classes

    @staticmethod
    def predict(frame):
        """
        predict scnene in the frame
        :param frame: input image as numpy array
        :return: class name of scene of the frame
        """
        model, image_transformer, classes = SceneClassifier.load_files()
        # convert to PIL image
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        frame = V(image_transformer(frame).unsqueeze(0))
        forward_pass = model.forward(frame)
        eval = F.softmax(forward_pass, 1).data.squeeze()
        probability, idx = eval.sort(0, True)
        return classes[idx[0]].replace('_', ' ')
