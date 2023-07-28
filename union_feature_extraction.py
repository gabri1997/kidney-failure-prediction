import torch
from torch import nn
from torchvision import transforms
from torchvision import models
import yaml
from yaml import CLoader as Loader
from torch import stack
from torch.utils.data.sampler import Sampler
import glob
import pandas as pd
import csv
import sklearn
from sklearn import metrics 
import numpy as np
from PIL import Image
import random
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics 


class MyResnet(nn.Module):
    def __init__(self, net='resnet18', pretrained=True, num_classes=1, dropout_flag=True):
        super(MyResnet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'resnet18':
            resnet = models.resnet18(pretrained)
            bl_exp = 1
        elif net == 'resnet34':
            resnet = models.resnet34(pretrained)
            bl_exp = 1
        elif net == 'resnet50':
            resnet = models.resnet50(pretrained)
            bl_exp = 4
        elif net == 'resnet101':
            resnet = models.resnet101(pretrained)
            bl_exp = 4
        elif net == 'resnet152':
            resnet = models.resnet152(pretrained)
            bl_exp = 4
        else:
            raise Warning("Wrong Net Name!!")

        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.2)
        n_features = 512 * bl_exp * 2
        self.first_fc = nn.Sequential(nn.Linear(n_features, n_features * 2),
                                      nn.BatchNorm1d(num_features=n_features * 2),
                                      nn.ReLU(inplace=True))
        self.second_fc = nn.Sequential(nn.Linear(n_features * 2, n_features * 2),
                                       nn.BatchNorm1d(num_features=n_features * 2),
                                       nn.ReLU(inplace=True))
        self.last_fc = nn.Linear(n_features * 2, num_classes)

    
    def forward(self, x):
        batch_size, input_patches = x.size(0), x.size(1)
        # to 2D
        # x = self.to_2D(x)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        #x=torch.stack([x,x,x],1)
        print(x.shape)

        x = self.resnet(x.float())
        # to bio
        # x = self.to_bio(x)
        x = x.view(batch_size, input_patches, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4)
        avg_x = self.avgpool(x)
        max_x = self.maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        x = self.first_fc(x)
        x = self.second_fc(x)
        if self.dropout_flag:
            x = self.dropout(x)
        #x = self.last_fc(x)
        return x
    
class FusedDataset(Dataset):

    def __init__(self, dataset=None, patches_per_bio=None, transforms_wsi=None, transforms_fluo=None, transforms_norm=None, split=None):
        

        self.years = 5
        self.patches_per_bio = patches_per_bio
        self.dataset = dataset
        self.transforms_wsi = transforms_wsi
        self.transforms_fluo=transforms_fluo
        self.transforms_norm=transforms_norm
        self.bios_wsi = {}
        self.bios_fluo = {}
        self.imgs_root_wsi = '/nas/softechict-nas-2/nefrologia/patches_dataset/images/'
        self.imgs_root_fluo = '/nas/softechict-nas-1/fpollastri/data/istologia/images/'
        self.split = split
        count=0
        #FLUO
        df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/HandFLUO-Bio-IMG-Person.xlsx", header = 0)
        all_images = glob.glob(self.imgs_root_fluo + '*.tif')
        #Legge il file yaml dove dentro ci sono tutte le info del dataset e lo mette nella variabile d
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)
        #WSI
        #all images sara una lista con path img1, path img2...
        all_images = glob.glob(self.imgs_root_wsi + '*.png')
        #Legge il file yaml dove dentro ci sono tutte le info del dataset e lo mette nella variabile d
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                    img_bio = d['bios'][i]['bio']
            
                    #PER MTB_D_MDB DEVO USARE IL TIPO STRINGA, PER GLI ALTRI INT
                    #id_images = df_fluo[df_fluo['Biopsia n.'] == img_bio]

                    id_images = df_fluo[df_fluo['Biopsia n.'] == int(img_bio)]
                    id_images_fluo = id_images[id_images['Type']=='IgA']['Id_image']

                    #creo i due percorsi

                    #WSI
                    imgs_path_wsi = [img for img in all_images if f'_{img_bio}_pas' in img]
            
                    #FLUO
                    image_path_fluo=[]
                    for id in id_images_fluo:
                        id_image = '/nas/softechict-nas-1/fpollastri/data/istologia/images/'+ id
                        image_path_fluo.append(id_image)
                        

                    #controllo che siano entrambi non vuoti

                    if imgs_path_wsi == [] or image_path_fluo ==[]:
                        print(f'bio {img_bio} has no both images')
                        continue
                    else:
                        count+=1
                    
                    #la label che calcolo è la stessa per entrambi
                    
                    img_esrd = d['bios'][i]['ESRD']
                    img_fup = float(d['bios'][i]['fup'])
                    img_lbl = 0
                    if img_esrd == 'FALSE':
                        img_lbl = 0.5 - min(img_fup, self.years) / (2. * self.years)
                    elif img_fup < (2. * self.years):
                        img_lbl = 0.5 + ((2. * self.years) - max(img_fup, self.years)) / (2. * self.years)

                    #creo 2 dizionari separati, uno per wsi e uno per fluo
                    self.bios_wsi[img_bio] = {'images': imgs_path_wsi, 'label': img_lbl}
                    self.bios_fluo[img_bio] = {'images': image_path_fluo, 'label': img_lbl}
                    

        #DIZIONARIO
        d_wsi = csv.writer(open("/homes/grosati/Medical/Unione_dizionario_WSI.csv", "w"))
        d_fluo = csv.writer(open("/homes/grosati/Medical/Unione_dizionario_FLUO.csv", "w"))
        # loop over dictionary keys and values
        for key, val in self.bios_wsi.items():

            # write every key and value to file
            d_wsi.writerow([key, val])
            
        for key, val in self.bios_fluo.items():

            # write every key and value to file
            d_fluo.writerow([key,val])
        

    def __getitem__(self, index):

            #print(index)
            bio_wsi = self.bios_wsi[list(self.bios_wsi.keys())[index]]
            bio_fluo= self.bios_fluo[list(self.bios_fluo.keys())[index]]

            try:
                #WSI
                #patches_wsi = len(bio_wsi['images'])
                #patches_wsi = random.sample(bio_wsi['images'], patches_wsi)
                patches_wsi = bio_wsi['images']

                #FLUO
                #patches_wsi = random.sample(bio_wsi['images'], self.patches_per_bio)
                patches_fluo = bio_fluo['images']
            except ValueError:

                print("Value error in get_item function")
                
                #patches_fluo = bio_fluo['images']
                #patches_fluo += [random.choice(bio_fluo['images']) for _ in range(self.patches_per_bio - len(bio_fluo['images']))]
               

            #ground è lo stesso per tutti
            ground= bio_wsi['label']
            images_wsi = []
            images_fluo = []

            for patch_fluo in patches_fluo:

                image_fluo = Image.open(patch_fluo)

            
                #FLUO
                if self.transforms_fluo is not None:
                    image_fluo = self.transforms_fluo(image_fluo)
                #DA USARE PER NORMALIZZAZIONE
                if image_fluo.shape == torch.Size([1, 772, 1040]):
                    image_fluo=torch.cat([image_fluo,image_fluo,image_fluo],0)

                if self.transforms_norm is not None:
                    image_fluo=image_fluo.float()
                    image_fluo = self.transforms_norm(image_fluo)

                images_fluo.append(image_fluo)

            for patch_wsi in patches_wsi:
              
                image_wsi = Image.open(patch_wsi)

                #WSI
                if self.transforms_wsi is not None:
                    image_wsi = self.transforms_wsi(image_wsi)
                
                images_wsi.append(image_wsi)
                

                #print(len(images_wsi))
                #print(len(images_fluo))

            return stack(images_wsi),stack(images_fluo),ground


    def __len__(self):
        return len(self.bios_fluo.keys())


if __name__ == "__main__":

    dname = '/nas/softechict-nas-2/nefrologia/patches_dataset/big_nephro_5Y_bios_dataset.yml'
    
    dataset_mean = (0.813, 0.766, 0.837)
    dataset_std = (0.148, 0.188, 0.124)
    
    dataset_mean_fluo_IgA_hand = (90.14926028, 90.14957433, 90.1492549)
    dataset_std_fluo_IgA_hand = (285.92956238, 285.92946316, 285.92956381)
    #preprocess_fn = transforms.RandomResizedCrop(size=(256, 512), scale=(.5, 1.0), ratio=(2., 2.))

    dictionary= {}

    inference_transforms_fluo = transforms.Compose([       
        transforms.ToTensor(), 
        transforms.Resize((772, 1040)), 
        ])
    
    fluo_norm = transforms.Compose([
        transforms.Normalize(dataset_mean_fluo_IgA_hand, dataset_std_fluo_IgA_hand),
        ])
    
    inference_transforms = transforms.Compose([    
        #preprocess_fn,
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
        ])
    

    ppb=4

    fused_dataset = FusedDataset(dataset=dname, patches_per_bio=ppb,transforms_fluo=inference_transforms_fluo,transforms_wsi=inference_transforms,transforms_norm= fluo_norm, split=['test'])
    #out = fused_dataset[0]
    
    fused_loader = DataLoader(fused_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             # drop_last=True,
                             pin_memory=False)

    model = MyResnet(net='resnet18',num_classes=1).to('cuda')
    model_fluo = MyResnet(net='resnet18', num_classes=1).to('cuda')

    model_fluo.eval()
    model.eval()

    with torch.no_grad():

        for i, (stack_wsi, stack_fluo, target) in enumerate (fused_loader):

            stack_wsi=stack_wsi.to('cuda')
            stack_fluo=stack_fluo.to('cuda')
            features_wsi = torch.squeeze(model(stack_wsi))
            features_fluo= torch.squeeze(model_fluo(stack_fluo))

            dictionary[i] = {"features_wsi":features_wsi, "feature_fluo":features_fluo, "label":target}

        #torch.save(dictionary,'features_train.pt')
        torch.save(dictionary,'features_test.pt')
        #d_wsi = csv.writer(open("/homes/grosati/Medical/Unione_dizionario_features_train.csv", "w"))
        #or key, val in dictionary.items():

            # write every key and value to file
            #d_wsi.writerow([key, val])