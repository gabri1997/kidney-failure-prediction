import sklearn
import torch
from argparse import ArgumentParser
import os
import torchvision
os.environ["OMP_NUM_THREADS"] = "1"
from torchvision import models
from torch import nn
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from big_nephro_dataset import YAML10YBiosDataset
from big_nephro_dataset import YAML10YBiosDatasetFluo
from big_nephro_dataset import YAML10YBiosDatasetAllPpb
from big_nephro_dataset import YAML10YDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import torch
import shutil
import itertools
from sklearn.metrics import ConfusionMatrixDisplay
import wandb
import random
from torch.utils.data import TensorDataset, DataLoader


class Features_loader():
    
    def load_index_features(self,index,features):
        x = features[list(features.keys())[index]]
        feature_wsi = x['features_wsi']
        feature_fluo = x['feature_fluo']
        label = x['label']
        #print(label)
        return (feature_wsi,feature_fluo,label)



class LinearClassifier(torch.nn.Module):

  def __init__(self, input_dim=4096, output_dim=1):
    super(LinearClassifier, self).__init__()
    self.hidden = nn.Linear(input_dim, 128)
    self.relu = nn.ReLU()
    self.linear = torch.nn.Linear(128,1)
    #self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.relu(self.hidden(x))
    x = self.linear(x)
    #x = self.sigmoid(x)
    return x

class MyNetwork():

  def __init__(self,  net, num_epochs, optimizer, criterion):
      self.net = net
      self.num_epochs = num_epochs
      self.model = model
      self.optimizer=optimizer
      self.criterion=criterion
  
  def train(self):
    
    for epoch in range(self.num_epochs):

        # Forward pass
        self.model.train()
        losses = []
        #sigm = nn.Sigmoid()
        #sofmx = nn.Softmax(dim=1)
        features = torch.load('/homes/grosati/Medical/features_train.pt')
        f_loader = Features_loader()
        tensor_x = []
        tensor_y = []

        for index in range(len(features)):

          f_wsi,f_fluo,lbl = f_loader.load_index_features(index,features)        
          union = torch.cat((f_wsi,f_fluo),0)

          tensor_x.append(union)
          tensor_y.append(lbl)

        tensor_x = torch.stack(tensor_x)
        tensor_y = torch.stack(tensor_y)
        #print(type(tensor_x))
        #print(type(tensor_y))
      
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=4)

        for feature,label in my_dataloader:
            
            #featore.shape = [4,4096]
            feature = feature.to('cuda')
            label = label.to('cuda', torch.float)
            output = self.model(feature)       
            #check_output = sigm(output)
            label = (label == 1.).float()
            loss = self.criterion(output,label)
            losses.append(loss.item())
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Print the loss every epoch
        print('Epoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)))

        '''
        for index in range(len(features)):
            
            feature_wsi,feature_fluo,label = f_loader.load_index_features(index,features)
            #print("Feature_wsi_shape:{0} ".format(feature_wsi.shape))
            #print("Feature_fluo_shape:{0} ".format(feature_fluo.shape))
            #print("Label:{0} ".format(label))
            union_features = torch.cat ((feature_wsi,feature_fluo),0)
            #print(union_features.shape)
            union_features = union_features.to('cuda')
            label = label.to('cuda', torch.float)

            output = self.model(union_features)       
            #check_output = sigm(output)
            label = (label == 1.).float()
            #label = label.unsqueeze(1)
            #print(output.size(), label.size())
            loss = self.criterion(output,label)
            losses.append(loss.item())
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Print the loss every epoch
        print('Epoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)))
      '''
  def eval(self):
      
      with torch.no_grad():
        
        features = torch.load('/homes/grosati/Medical/features_test.pt')
        preds = np.zeros(len(features))
        gts = np.zeros(len(features))
        thresh=0.5
        sigm = nn.Sigmoid()
        self.model.eval()
        #print(len(features))
        f_loader = Features_loader()

        for index in range(len(features)):
          
              feature_wsi,feature_fluo,label = f_loader.load_index_features(index,features)
              #print("Feature_wsi_shape:{0} ".format(feature_wsi.shape))
              #print("Feature_fluo_shape:{0} ".format(feature_fluo.shape))
              #print("Label:{0} ".format(label))
              union_features = torch.cat ((feature_wsi,feature_fluo),0)
              #print(union_features.shape)
              union_features = union_features.to('cuda')
              label = label.to('cuda', torch.float)

              output = self.model(union_features)    
              #print("L'output è {0}".format(output))   
              check_output = sigm(output)
              #print("L'output è check output {0}".format(check_output))
              label = (label == 1.).float()
              #print("La label è {0}".format(label))
              #label = label.unsqueeze(1)
              #print(output.size(), label.size())
              gts[index] = label.to('cpu')
              preds[index] = check_output.to('cpu')
        print(gts)
        print(preds)

        #Metrics
        bin_preds = np.where(preds > thresh, 1., 0.)    
        tr_targets = gts * 2 - 1    
        trues = sum(bin_preds)
        tr_trues = sum(bin_preds == tr_targets)
        g_trues = sum(gts) 
        pr = tr_trues / (trues + 10e-5)
        rec = tr_trues / g_trues
        spec = (sum(gts == bin_preds) - tr_trues) / sum(gts == 0)
        fscore = (2 * pr * rec) / (pr + rec + 10e-5)
        acc = np.mean(gts == bin_preds).item()
        auc = metrics.roc_auc_score(gts, preds)
        stats_string = f'Acc: {acc:.3f} | AUC: {auc:.3f} | F1 Score: {fscore:.3f} | Precision: {pr:.3f} | Recall: {rec:.3f} | Specificity: {spec:.3f} | Trues: {trues:.0f} | Correct Trues: {tr_trues:.0f} | ' \
                                f'Ground Truth Trues: {g_trues:.0f}'
        confusion_matrix = metrics.confusion_matrix(gts, bin_preds) 
        print("This is conf_matrix:\n{0}".format(confusion_matrix))
        print(stats_string)

        #print(gts)
        #print(preds)    


if __name__ == '__main__':
     

    model = LinearClassifier().to('cuda')
    # Train the model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    #bce_weight = torch.tensor([15.]).cuda()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = torch.nn.BCELoss()
  
    union_classifier = MyNetwork(net=model,num_epochs=200,optimizer=optimizer,criterion=criterion)
    
    union_classifier.train()
    union_classifier.eval()

        
    
    
    
    
          
      
