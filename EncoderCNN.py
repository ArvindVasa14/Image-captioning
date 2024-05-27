import torch
from torch import nn
import numpy as np
from torchvision import models
import torchvision

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN= False):
        super(EncoderCNN, self).__init__()
        self.embed_size= embed_size
        self.train_CNN= train_CNN
        self.inception= models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        self.inception.fc= nn.Linear(self.inception.fc.in_features, self.embed_size)
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(0.5)


    def forward(self, images):
        features, _ = self.inception(images)
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad= True
            else:
                param.requires_grad= self.train_CNN
        return self.dropout(self.relu(features))


if __name__=="__main__":
    cnn = EncoderCNN(500)
    t = torch.randint(0, 256, (2, 3, 299, 299))
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    print(t.shape)
    # Normalize the tensor
    # t = (t / 255.0 - mean[:, None, None]) / std[:, None, None]
    print(cnn(t))
