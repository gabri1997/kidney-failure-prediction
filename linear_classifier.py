import torch
from argparse import ArgumentParser
import os
os.environ["OMP_NUM_THREADS"] = "1"
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import torch
from torch.utils.data import TensorDataset, DataLoader


class Features_loader():
    
    def load_index_features(self,index,features):
        x = features[list(features.keys())[index]]
        feature_wsi = x['features_wsi']
        feature_fluo = x['feature_fluo']
        label = x['label']
        #print(label)
        return (feature_wsi,feature_fluo,label)

    def load_index_features_wsi(self,index,features):
        x = features[list(features.keys())[index]]
        feature_wsi = x['features_wsi']
        label = x['label']
        #print(label)
        return (feature_wsi,label)

class LinearClassifier(torch.nn.Module):

  def __init__(self, input_dim):
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
      self.criterion= criterion
      self.optimizer=optimizer
  
  def train_fused(self, path_features_train_fused):
    
    for epoch in range(self.num_epochs):

        # Forward pass
        self.model.train()
        losses = []
        #sigm = nn.Sigmoid()
        #sofmx = nn.Softmax(dim=1)
        features = torch.load(path_features_train_fused)
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

        #no batch
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
  def eval_fused(self, path_features_test_fused):
      
      with torch.no_grad():
        
        features = torch.load(path_features_test_fused)
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
  
  def train_wsi(self,path_features_train_wsi):
     
     for epoch in range(self.num_epochs):

        # Forward pass
        self.model.train()
        losses = []
        sigm = nn.Sigmoid()
        #sofmx = nn.Softmax(dim=1)
        features = torch.load(path_features_train_wsi)
        f_loader = Features_loader()
       
        batch_features = []
        batch_labels = []
        for index in range(len(features)):
          feature, label = f_loader.load_index_features_wsi(index, features)
          batch_features.append(feature)
          batch_labels.append(label)
        batch_features = torch.stack(batch_features)
        batch_labels = torch.stack(batch_labels)

        for index in range(len(features)):

          # f_wsi,label = f_loader.load_index_features_wsi(index,features) 

          f_wsi = batch_features.to('cuda')
          label = batch_labels.to('cuda', torch.float)

          output = self.model(f_wsi)       
          check_output = sigm(output)
          label = (label == 1.).float()
          loss = self.criterion(check_output,label)
          losses.append(loss.item())
          # Backward pass and optimization
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
         
        print('Epoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)))
        
  def eval_wsi(self,path_features_test_wsi):
     
     with torch.no_grad():
        
        features = torch.load(path_features_test_wsi)
        preds = np.zeros(len(features))
        gts = np.zeros(len(features))
        thresh=0.5
        sigm = nn.Sigmoid()
        self.model.eval()
        #print(len(features))
        f_loader = Features_loader()

        for index in range(len(features)):
          
              feature_wsi,label = f_loader.load_index_features_wsi(index,features)
              #print("Feature_wsi_shape:{0} ".format(feature_wsi.shape))
              #print("Feature_fluo_shape:{0} ".format(feature_fluo.shape))
              #print("Label:{0} ".format(label))
              #print(union_features.shape)
              feature_wsi = feature_wsi.to('cuda')
              label = label.to('cuda', torch.float)

              output = self.model(feature_wsi)    
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
        if self.num_epochs == 0:
                threshes = np.arange(100) / 100.0
        else:
                threshes = [thresh]
        for t in threshes:
                print(f'\nthresh: {t}')
                bin_preds = np.where(preds > t, 1., 0.)
                tr_targets = gts * 2 - 1
                trues = sum(bin_preds)
                y_true = gts
                y_pred = bin_preds
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
                print(stats_string)
                confusion_matrix = metrics.confusion_matrix(gts, bin_preds) 
                print("This is conf_matrix:\n{0}".format(confusion_matrix))
        return auc, fscore, rec, spec, pr, y_true, y_pred


if __name__ == '__main__':
     
    path_features_train_fused = '/homes/grosati/Medical/features_train.pt'
    path_features_test_fused = '/homes/grosati/Medical/features_test.pt'
    path_features_train_wsi = '/homes/grosati/Medical/features_train_wsi.pt'
    path_features_test_wsi = '/homes/grosati/Medical/features_test_wsi.pt'
    
    parser = ArgumentParser()
    parser.add_argument('--type', default='fused')
    opt = parser.parse_args()
    print(opt.type)
    

    if opt.type == "fused":   
      model = LinearClassifier(input_dim=4096).to('cuda')   
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
      #bce_weight = torch.tensor([15.]).cuda()
      #criterion = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
      criterion = nn.BCEWithLogitsLoss()
      #criterion = torch.nn.BCELoss()
      union_classifier = MyNetwork(net=model,num_epochs=200,optimizer=optimizer,criterion=criterion)
      # Train the model
      union_classifier.train_fused(path_features_train_fused)
      # Test the model
      union_classifier.eval_fused(path_features_test_fused)

    if opt.type == "wsi":
      weights = torch.tensor([10.]).cuda()
      model = LinearClassifier(input_dim=2048).to('cuda')  
      opt = torch.optim.SGD(model.parameters(), lr=0.1)
      #pos_weight=weights
      loss = nn.BCEWithLogitsLoss()
      classifier_wsi = MyNetwork(net=model,num_epochs=60,optimizer=opt,criterion=loss)
      # Train the model
      classifier_wsi.train_wsi(path_features_train_wsi)
      # Test the model
      classifier_wsi.eval_wsi(path_features_test_wsi)

        
    
    #WSI (BIO) IN TRAIN SONO 338 (0) 25 (1) 
    #WSI (BIO) IN TEST SONO 104 (0) 29 (1) 
    
    
          
      
