from datetime import datetime

from easydict import EasyDict
import torch
import torchvision
import matplotlib.pyplot as plt

from model.cnn import Classifier
from model.deconv import Deconv
from conf.config import CONF

def visualize_net():
    # set seeds for reproducability
    torch.manual_seed(CONF.TRAIN.MANUAL_SEED)
    torch.cuda.manual_seed_all(CONF.TRAIN.MANUAL_SEED)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(degrees=(90, 90)),
        torchvision.transforms.RandomVerticalFlip(1),
        torchvision.transforms.ToTensor(),
    ])

    # load testing dataset
    emnist = torchvision.datasets.EMNIST("data", split="byclass", download=True, transform=transforms, train=False)

    # load model
    model = Deconv(CONF.MODEL)
    model.eval()

    #CUDA stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # for now just take the second picture and the second block for starting the deconv net
    pic = emnist[3][0].unsqueeze(0)
    visualization = model(pic, CONF.DECONV.N)
    
    plt.imshow(visualization.squeeze().detach())



visualize_net()