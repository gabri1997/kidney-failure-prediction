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
import numpy as np
from PIL import Image
import random
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser


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
            #print("La shape di x: {0} ".format(x.shape))
            #La shape di x: torch.Size([8, 3, 772, 1040])
            #estrae feature con resnet
        x = self.resnet(x.float())
            #print("La shape di x dopo resent float: {0} ".format(x.shape))
            #La shape di x dopo resent float: torch.Size([8, 512, 25, 33]) 
        x = x.view(batch_size, input_patches, x.size(1), x.size(2), x.size(3))
            #print("La shape di x prima di permutare: {0} ".format(x.shape))
            #La shape di x prima di permutare: torch.Size([2, 4, 512, 25, 33]) 
        x = x.permute(0, 2, 1, 3, 4)
            #print("La shape di x dopo la permutazione: {0} ".format(x.shape))
            #La shape di x dopo la permutazione: torch.Size([2, 512, 4, 25, 33]) 

        #print ("Questa è la shape prima {0}".format(x.shape))
        avg_x = self.avgpool(x)
        #print ("Questa è la shape dopo avg {0}".format(avg_x.shape))
        max_x = self.maxpool(x)
        #print ("Questa è la shape dopo max_x {0}".format(max_x.shape))
            
        x = torch.cat((avg_x, max_x), dim=1)
            #print("La shape di x dopo cat: {0} ".format(x.shape))
            #La shape di x dopo cat: torch.Size([2, 1024, 1, 1, 1])
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
            #print("La shape di x prima del forward: {0} ".format(x.shape))
            #La shape di x prima del forward: torch.Size([2, 1024]) 
        x = self.first_fc(x)
            #print("La shape di x dopo first_fc: {0} ".format(x.shape))
            #La shape di x dopo second_fc: torch.Size([2, 2048])
        x = self.second_fc(x)
            #print("La shape di x dopo second_fc: {0} ".format(x.shape))
            #La shape di x dopo second_fc: torch.Size([2, 2048])
        if self.dropout_flag:
            x = self.dropout(x)
        #x = self.last_fc(x)
        return x
    
class YAML10YBiosDataset(Dataset):
    # PATCHES mean values BGR = [0.81341412 0.76660304 0.83704776] | std values BGR = [0.14812355 0.18829341 0.12363736]
    # [0.74629832 0.67295842 0.78365591] | std values RGB = [0.17482606 0.21674619 0.14285819]
    #def __init__(self, years, dataset, crop_type, patches_per_bio, transforms=None, split=['training']):
    def __init__(self, dataset, patches_per_bio, transforms=None, split=['training']):
        """
           Initializes a pytorch Dataset object
           :param dataset: A filename (string), to identify the yaml file
              containing the dataset.
           :param transform: Transformation function to be applied to the input
              images (e.g. created with torchvision.transforms.Compose()).
           :param split: A list of strings, one for each dataset split to be
              loaded by the Dataset object.
           """

        self.years = 5
        self.patches_per_bio = patches_per_bio
        self.dataset = dataset
        self.transforms = transforms
        self.bios = {}
        #self.imgs_root = os.path.join(os.path.dirname(dataset), crop_type + '_images/')
        self.imgs_root = '/nas/softechict-nas-2/nefrologia/patches_dataset/images/'
        self.split = split
        print("SPLIT")
        print(self.split)
        #all images sara una lista con path img1, path img2...
        all_images = glob.glob(self.imgs_root + '*.png')
        #Legge il file yaml dove dentro ci sono tutte le info del dataset e lo mette nella variabile d
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

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
                    img_lbl = 0.5 - min(img_fup, self.years) / (2. * self.years)
                elif img_fup < (2. * self.years):
                    img_lbl = 0.5 + ((2. * self.years) - max(img_fup, self.years)) / (2. * self.years)
            
                #alla fine avro questo dizionario dove dentro avro il nome dell immagine, il path e la label
                self.bios[img_bio] = {'images': imgs_path, 'label': img_lbl}
                #QUINDI IL DATALOADER VA A CARICARE IN MEMORIA IN QUALCHE MODO UN DIZIONARIO DOVE HO TUTTE LE INFO E I DATI
                

        #DIZIONARIO
        w = csv.writer(open("/homes/grosati/Medical/dizionario_label.csv", "w"))

        # loop over dictionary keys and values
        for key, val in self.bios.items():

            # write every key and value to file
            w.writerow([key, val])
  
    def __getitem__(self, index):
        bio = self.bios[list(self.bios.keys())[index]]
        #ppb=len(bio['images'])
        #print("Vediamo il numero di patches per questa wsi{0}: ".format(ppb))
        try:
            #OGNI IMMAGINE HA DIVERSE PATCH ASSOCIATE E QUI IO VADO A PRENDERE UN SAMPLE RANDOMICO DI ALCUNE PATCHES
            #RANDOM SAMPLING WITHOUT REPLACEMENT
            patches = random.sample(bio['images'], self.patches_per_bio)
        except ValueError:
            patches = bio['images']
            patches += [random.choice(bio['images']) for _ in range(self.patches_per_bio - len(bio['images']))]
            random.shuffle(patches)
        
        ground = bio['label']
        images = []

        for patch in patches:
       
            image = Image.open(patch)
            # debug_plot(np.array(image))
            if self.transforms is not None:
                image = self.transforms(image)
                
            # debug_plot(np.array(image))
            images.append(image)

        #Qui ritorna un batch con tutte quelle patch una sopra all'altra
        #print("The index is{0}, the label is{1}".format(index,ground))
        return stack(images), ground

    def __len__(self):
        return len(self.bios.keys())
class YAML10YBiosDatasetAllPpb(Dataset):
    # PATCHES mean values BGR = [0.81341412 0.76660304 0.83704776] | std values BGR = [0.14812355 0.18829341 0.12363736]
    # [0.74629832 0.67295842 0.78365591] | std values RGB = [0.17482606 0.21674619 0.14285819]
    def __init__(self, dataset, crop_type, patches_per_bio, transforms=None, split=['training']):
    
        self.years = 5
        self.patches_per_bio = patches_per_bio
        self.dataset = dataset
        self.transforms = transforms
        self.bios = {}
        self.imgs_root = '/nas/softechict-nas-2/nefrologia/patches_dataset/images/'
        self.split = split
        print("SPLIT")
        print(self.split)
        #all images sara una lista con path img1, path img2...
        all_images = glob.glob(self.imgs_root + '*.png')
        #Legge il file yaml dove dentro ci sono tutte le info del dataset e lo mette nella variabile d
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)
        count=0
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
                    img_lbl = 0.5 - min(img_fup, self.years) / (2. * self.years)
                elif img_fup < (2. * self.years):
                    img_lbl = 0.5 + ((2. * self.years) - max(img_fup, self.years)) / (2. * self.years)
            
                #alla fine avro questo dizionario dove dentro avro il nome dell immagine, il path e la label
                self.bios[img_bio] = {'images': imgs_path, 'label': img_lbl}
                for s in imgs_path:
                    count += 1
        print("Questo è il numero di immagini in split:{0}".format(split))  
        print(count) 

        #DIZIONARIO
        w = csv.writer(open("/homes/grosati/Medical/dizionario_label.csv", "w"))

        # loop over dictionary keys and values
        for key, val in self.bios.items():

            # write every key and value to file
            w.writerow([key, val])
  
    def __getitem__(self, index):

        bio = self.bios[list(self.bios.keys())[index]]
        #ppb_wsi = len(bio['images'])
        #patches = random.sample(bio['images'], ppb_wsi)
        patches = bio['images']
        
        ground = bio['label']
        images = []

        for patch in patches:
            
            #Carico le immagini una ad una
            image = Image.open(patch)
          
            if self.transforms is not None:
                image = self.transforms(image)
           
            images.append(image)

        #Qui ritorna un batch con tutte quelle patch una sopra all'altra
        return stack(images), ground
    
    def __len__(self):
        return len(self.bios.keys())

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--split', default='training')
    opt = parser.parse_args()
    print(opt.split)

    dname = '/nas/softechict-nas-2/nefrologia/patches_dataset/big_nephro_5Y_bios_dataset.yml'
    
    dataset_mean = (0.813, 0.766, 0.837)
    dataset_std = (0.148, 0.188, 0.124)
    
  
    preprocess_fn = transforms.RandomResizedCrop(size=(256, 512), scale=(.5, 1.0), ratio=(2., 2.))

    dictionary= {}

   
    inference_transforms = transforms.Compose([    
        #preprocess_fn,
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
        ])
    custom_training_transforms = transforms.Compose([
              
                transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(180, fill=255)]), p=.25),
                preprocess_fn,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(contrast=(0.5, 1.7)),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std),
            ])
    

    ppb=4

    if opt.split == "training":
        #dataset = YAML10YBiosDataset(dataset=dname, patches_per_bio=ppb, transforms=custom_training_transforms, split=['training'])
        dataset = YAML10YBiosDatasetAllPpb(dataset=dname, crop_type=None,patches_per_bio=ppb, transforms=custom_training_transforms, split=['training'])
        data_loader = DataLoader(dataset,batch_size=1,shuffle=True,drop_last=True,pin_memory=False)
    else:
        #dataset = YAML10YBiosDataset(dataset=dname, patches_per_bio=max(16, ppb * 2), transforms=inference_transforms, split=['test'])
        dataset = YAML10YBiosDatasetAllPpb(dataset=dname, crop_type=None, patches_per_bio=ppb, transforms=custom_training_transforms, split=['test'])
        data_loader = DataLoader(dataset,batch_size=1,shuffle=False,drop_last=False,pin_memory=False)
       
    model = MyResnet(net='resnet18',num_classes=1, dropout_flag=False).to('cuda')
    

    path_wsi = "/homes/grosati/Medical/MODELS/resnet18_5Y_Z2_TEST_WSI_AllPPB_8BATCH_4PPBTRAIN_Random_net.pth"
    model.load_state_dict(torch.load(path_wsi))

    model.eval()

    with torch.no_grad():

        for i, (stack_wsi, target) in enumerate (data_loader):

            stack_wsi=stack_wsi.to('cuda')
            features_wsi = torch.squeeze(model(stack_wsi))
            
            print("Shape:",features_wsi.shape)
            dictionary[i] = {"features_wsi":features_wsi, "label":target}
        
        
        if opt.split == "training":
            torch.save(dictionary,'features_train_wsi.pt')
            df = pd.DataFrame(dictionary)
            df_transposed = df.T
            output_file = 'features_train_wsi.xlsx'
            df_transposed.to_excel(output_file, index=False)  # Set index=False to exclude the index column

        else:
            df = pd.DataFrame(dictionary)
            df_transposed = df.T
            output_file = 'features_test_wsi.xlsx'
            torch.save(dictionary,'features_test_wsi.pt')
            df_transposed.to_excel(output_file, index=False)
   


    