import torch 
import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size): 
        super(EncoderCNN, self).__init__() # constructor of parent class 

        rn50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in rn50.parameters():
            param.requires_grad = False

        # remvoe the last layer - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        modules = list(rn50.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # linear layer for projection: input_size = in_features, projected to embedding layer size 
        self.embed = nn.Linear(rn50.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.relu = nn.ReLU()
        
    def forward(self, images):
        with torch.no_grad():

            features = self.resnet(images) # [batch_size, # feature channels, h, w]
            
            features = features.reshape(features.size(0), -1) # flattens the features to create a single vector per image - [batch_size, # feature channels]
        
        features = self.embed(features)
        features = self.bn(features)
        features = self.relu(features)
        
        return features