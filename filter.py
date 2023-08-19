import numpy as np
from collections import Counter
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Sampler
import random
import big_nephro_dataset
import sklearn
import torch
import glob
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
from sklearn import metrics
import torch
from sklearn.metrics import ConfusionMatrixDisplay
import wandb
import random
from torch.utils.data.sampler import SubsetRandomSampler
import yaml
from yaml import CLoader as Loader


class filter_class():
    
    #not hard coded
    #bisogna solo cambiare la root a seconda della tipologia di esperimento, questo è all ppb
    def auto_filter_list (dataset,imgs_root):

        years = 5
        bios = {}
        all_images = glob.glob(imgs_root + '*.png')
        split = ['unique']
        with open(dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)
        
        list_label_0 = []
        list_label_1 = []
        #split train test e eval, prendo tutte le immagini dentro quello split li
        for s in split:
            for i in d['split'][s]:
                img_bio = d['bios'][i]['bio']
                # imgs_path = glob.glob(self.imgs_root + f'id*_{img_bio}*.png')
                imgs_path = [img for img in all_images if f'_{img_bio}_pas' in img]
                if imgs_path == []:
                    print(f'bio {img_bio} has no images')
                    continue
                img_esrd = d['bios'][i]['ESRD']
                img_fup = float(d['bios'][i]['fup'])
                img_lbl = 0
                if img_esrd == 'FALSE':
                    img_lbl = 0.5 - min(img_fup, years) / (2. * years)
                elif img_fup < (2. * years):
                    img_lbl = 0.5 + ((2. * years) - max(img_fup, years)) / (2. * years)
                
                if (img_lbl == 1):
                    list_label_1.append(i)
                else :
                    list_label_0.append(i)

                #alla fine avro questo dizionario dove dentro avro il nome dell immagine, il path e la label
                bios[img_bio] = {'images': imgs_path, 'label': img_lbl}
        
        # Using list comprehension
        filtered_list_0 = [item for item in list_label_0 if item not in list_label_1]
        # Using filter function and lambda
        filtered_list_0 = list(filter(lambda item: item not in list_label_1, list_label_0))
        print(len(filtered_list_0))
        print(filtered_list_0)
        print(len(list_label_1))
        print(list_label_1)

        return filtered_list_0,list_label_1

    def hard_coded_filter_list ():

        first_list  = list(range(496))
        second_list = [137,151,152,196,203,205,207,208,216,231,235,280,291,293,298,303,306,307,309,310,311,316,323,324,
                        16,23,25,35,39,48,50,51,52,55,57,60,68,71,77,81,85,88,102,104,105,110,112,114,115,122,126,127,132]
        # Using list comprehension
        filtered_list = [item for item in first_list if item not in second_list]
        # Using filter function and lambda
        filtered_list = list(filter(lambda item: item not in second_list, first_list))
        #print(len(filtered_list))

        return filtered_list,second_list

    def index_list_composer(num_samples_train_0, num_samples_test_0,num_samples_train_1,num_samples_test_1,l_0, l_1):

        random_samples_train = []
        random_samples_test = []

        for _ in range(num_samples_train_0):

            if l_0:
                picked_value_train = random.choice(l_0)
                random_samples_train.append(picked_value_train)
                l_0.remove(picked_value_train)
            

        for _ in range (num_samples_train_1):

            if l_1:
                picked_value_train = random.choice(l_1)
                random_samples_train.append(picked_value_train)
                l_1.remove(picked_value_train)
           
    

        for _ in range (num_samples_test_0):

            if l_0:
                picked_value_test = random.choice(l_0)
                random_samples_test.append(picked_value_test)
                l_0.remove(picked_value_test)
        
        for _ in range(num_samples_test_1):

            if l_1:
                picked_value_test = random.choice(l_1)
                random_samples_test.append(picked_value_test)
                l_1.remove(picked_value_test)

        return random_samples_train, random_samples_test

if __name__ == '__main__':

    dname= '/homes/grosati/Medical/big_nephro_5Y_bios_dataset_split.yml'
    root = '/nas/softechict-nas-2/nefrologia/patches_dataset/images/'

    list_label_0,list_label_1 = filter_class.auto_filter_list(dataset=dname,imgs_root=root)
    #print(list_label_0)
    #print(list_label_1)
    random_samples_train, random_samples_test = filter_class.index_list_composer(num_samples_train_0=250,
                                                                                 num_samples_test_0=100,
                                                                                 num_samples_train_1=30,
                                                                                 num_samples_test_1=20,
                                                                                 l_0 = list_label_0,l_1 =list_label_1)
      
    #print((random_samples_train))
    #print((random_samples_test))
                                                                       
    sampler_train = SubsetRandomSampler(random_samples_train)
    sampler_test = SubsetRandomSampler(random_samples_test)

    print(sampler_test)

    preprocess_fn = transforms.RandomResizedCrop(size=(256, 512), scale=(.5, 1.0), ratio=(2., 2.))   

    dataset_mean = (0.813, 0.766, 0.837)
    dataset_std = (0.148, 0.188, 0.124)
        
    custom_training_transforms = transforms.Compose([
              
                transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(180, fill=255)]), p=.25),
                preprocess_fn,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(contrast=(0.5, 1.7)),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std),
            ])
    inference_transforms = transforms.Compose([
               
                #preprocess_fn,
                transforms.Resize(size=(256, 512)),
             
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std),
            ])

    dataset = YAML10YBiosDataset(dataset=dname, crop_type='patches', patches_per_bio=4, transforms=custom_training_transforms, split=['unique'])
            
    test_dataset = YAML10YBiosDatasetAllPpb(dataset=dname, crop_type='patches', patches_per_bio=None, transforms=inference_transforms, split=['unique'])


    data_loader = DataLoader(dataset,batch_size=4,shuffle=False,drop_last=True,pin_memory=False,sampler = sampler_train)
    test_data_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,drop_last=True,pin_memory=False,sampler = sampler_test)

    #Essendo diviso in batch da 4 avrè 70 se stampo la lunghezza del data_loader, 120 è il test_data_loader
    #print (len(data_loader))
    
