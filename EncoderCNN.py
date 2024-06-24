import torch
from torch import nn
import numpy as np
from torchvision import models
import torchvision

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, train_CNN= False):
        super(EncoderCNN, self).__init__()
        self.embed_size= embed_size
        self.hidden_size= hidden_size
        self.train_CNN= train_CNN
        self.vgg= models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.vgg.classifier= nn.Sequential(
            nn.Linear(self.vgg.classifier[0].in_features, self.embed_size),
            nn.Linear(self.embed_size, self.hidden_size)
        )
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(0.5)
        for name, param in self.vgg.named_parameters():
            if "classifier" in name:
                param.requires_grad= True
            else:
                param.requires_grad= self.train_CNN


    def forward(self, images):
        features = self.vgg(images)
        return self.dropout(self.relu(features))

if __name__=="__main__":
    enc= EncoderCNN(256, 512)
    # print(enc)
    # for name, param in enc.named_parameters():
    #     print(name, param.size())
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(weights= "IMAGENET1K_V1")
#         # disable learning for parameters
#         for param in resnet.parameters():
#             param.requires_grad_(False)
#
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)
#
#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features