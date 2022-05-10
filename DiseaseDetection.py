import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable
import cv2



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location='cpu')
    model = models.resnet152()

    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 33

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 512)),
        ('relu', nn.ReLU()),
        # ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(512, 33)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']
#
#
# # Get index to class mapping
class_to_idx = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2,
                'Apple___healthy': 3, 'Cherry___Powdery_mildew': 4, 'Cherry___healthy': 5,
                'Corn___Cercospora_leaf_spot Gray_leaf_spot': 6, 'Corn___Common_rust': 7,
                'Corn___Northern_Leaf_Blight': 8, 'Corn___healthy': 9, 'Grape___Black_rot': 10,
                'Grape___Esca_(Black_Measles)': 11, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 12,
                'Grape___healthy': 13, 'Peach___Bacterial_spot': 14, 'Peach___healthy': 15,
                'Pepper,_bell___Bacterial_spot': 16, 'Pepper,_bell___healthy': 17, 'Potato___Early_blight': 18,
                'Potato___Late_blight': 19, 'Potato___healthy': 20, 'Strawberry___Leaf_scorch': 21,
                'Strawberry___healthy': 22, 'Tomato___Bacterial_spot': 23, 'Tomato___Early_blight': 24,
                'Tomato___Late_blight': 25, 'Tomato___Leaf_Mold': 26, 'Tomato___Septoria_leaf_spot': 27,
                'Tomato___Spider_mites Two-spotted_spider_mite': 28, 'Tomato___Target_Spot': 29,
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 30, 'Tomato___Tomato_mosaic_virus': 31,
                'Tomato___healthy': 32}
idx_to_class = {v: k for k, v in class_to_idx.items()}





def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage / 255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485) / (0.229)
    imgB = (imgB - 0.456) / (0.224)
    imgC = (imgC - 0.406) / (0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file

    image = torch.FloatTensor([process_image(image_path)])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1]
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

