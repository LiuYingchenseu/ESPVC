import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display

import warnings
warnings.filterwarnings('ignore')

from torchsummary import summary


#! cnn_transformer
class serial_cnn_transformer(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        
        # (1, 64, 251) -> (16, 32, 125) -> （32, 12, 62）-> (128, 4, 15) -> (256, 1, 3)
        ######################## CNN BLOCK #############################
        self.conv2Dblock1 = nn.Sequential(
            
            #! 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            #! 2nd 2D convolution layer
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            #! 3rd 2D convolution layer
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            #! 4th 2D convolution layer
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            #! 5th 2D convolution layer
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        
        ##################### TRANSFORMER BLOCK ##########################
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=768 + 2,
            nhead=5,
            dim_feedforward=1024, 
            dropout=0.4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        
        ################# FINAL LINEAR BLOCK ####################
        self.fc1_linear = nn.Linear(768 + 2, 64)
        self.classifier = nn.Linear(64, num_class)
        self.softmax_out = nn.Softmax(dim=1)
    
    
    def forward(self, x, gender, age):
        
        # CNN forward pass
        conv2d_embedding = self.conv2Dblock1(x)
        conv2d_embedding = torch.flatten(conv2d_embedding, start_dim=1)

        # Concatenate gender and age
        gender = gender.unsqueeze(1)
        age = age.unsqueeze(1)
        conv2d_embedding = torch.cat([conv2d_embedding, gender, age], dim=1)

        # Prepare for transformer input
        transformer_input = conv2d_embedding.unsqueeze(0)

        # Transformer forward pass
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_embedding = transformer_output.squeeze(0)  # Remove the batch dimension

        # Final linear layers
        output_1 = self.fc1_linear(transformer_embedding)
        output_logits = self.classifier(output_1)
        output_softmax = self.softmax_out(output_logits)

        return output_1, output_logits, output_softmax


#! resnet_transformer
class serial_resnet_transformer(nn.Module):
    
    def __init__(self, num_class, cnn_type):
        super().__init__()
        
        if cnn_type == "resnet18":
            pre_cnn = models.resnet18(pretrained=True)
        elif cnn_type == "resnet34":
            pre_cnn = models.resnet34(pretrained=True)
        elif cnn_type == "resnet50":
            pre_cnn = models.resnet50(pretrained=True)
        
        self.cnn = nn.Sequential(*list(pre_cnn.children())[:-2])
        self.cnn[0] = nn.Conv2d(1, self.cnn[0].out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.cnn_avgpool = pre_cnn.avgpool
        self.cnn_fc_in_features = pre_cnn.fc.in_features
        
        print(self.cnn_fc_in_features)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_fc_in_features + 2,
            nhead=2,
            dim_feedforward=1024, 
            dropout=0.4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        self.fc1_linear = nn.Linear(self.cnn_fc_in_features + 2, 64)
        self.classifier = nn.Linear(64, num_class)
        self.softmax_out = nn.Softmax(dim=1)
    
    def forward(self, x, gender, age):
        cnn_out = self.cnn(x)
        cnn_out = self.cnn_avgpool(cnn_out)
        cnn_embedding = torch.flatten(cnn_out, 1)
        
        gender = gender.unsqueeze(1)
        age = age.unsqueeze(1)
        conv2d_embedding = torch.cat([cnn_embedding, gender, age], dim=1)
        
        transformer_input = conv2d_embedding.unsqueeze(0)
        
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_embedding = transformer_output.squeeze(0)
        
        output_1 = self.fc1_linear(transformer_embedding)
        output_logits = self.classifier(output_1)
        output_softmax = self.softmax_out(output_logits)
        
        return output_1, output_logits, output_softmax