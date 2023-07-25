from PIL import Image
import os.path
import yaml
import cv2
from yaml import CLoader as Loader
# from rotated_rectangle_crop import crop_rotated_rectangle
from torch.nn import ModuleList
from torch import stack
import torch.utils.data as data
import numpy as np
import torch
import random
from torch.utils.data.sampler import Sampler
import glob
import glob
import csv
import numpy as np
import pandas as pd
from torchvision import transforms
from numpy import asarray
from collections import Counter

# mean values RGB = [0.60608787 0.57173514 0.61699724] | std values RGB = [0.37850211 0.37142419 0.38158805]
class YAMLSegmentationDataset(data.Dataset):

    def __init__(self, dataset=None, transforms=None, split=['training']):
        """
           Initializes a pytorch Dataset object
           :param dataset: A filename (string), to identify the yaml file
              containing the dataset.
           :param transform: Transformation function to be applied to the input
              images (e.g. created with torchvision.transforms.Compose()).
           :param split: A list of strings, one for each dataset split to be
              loaded by the Dataset object.
           """

        self.dataset = dataset
        self.transform = transforms
        self.imgs = []
        self.lbls = []

        self.imgs_root = os.path.join(os.path.dirname(dataset), 'patches_images/')
        print("root")
        print (self.imgs_roots)
        all_images = glob.glob(self.imgs_root + '*.png')
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            for i in d['split'][s]:
                img_bio = d['bios'][i]['bio']
                imgs_path = [img for img in all_images if f'_{img_bio}_pas' in img]

                self.imgs += imgs_path
                self.lbls += [img.replace('/patches_images/', '/patches_gts/').replace('.png', '_gt.npy') for img in imgs_path]

    def __getitem__(self, index):

        image = np.asarray(Image.open(self.imgs[index]))
        ground = np.load(self.lbls[index])

        if self.transform is not None:
            image, ground = self.transform(image, ground)

        return image, ground, os.path.basename(self.imgs[index])

    def __len__(self):
        return len(self.lbls)


class YAML10YDataset(data.Dataset):
    # mean values BGR = [0.81341412 0.76660304 0.83704776] | std values BGR = [0.14812355 0.18829341 0.12363736]
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

        self.patches_per_bio = patches_per_bio
        self.dataset = dataset
        self.transform = transforms
        self.bios = {}

        data_root = os.path.dirname(dataset)

        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        for s in split:
            # build a dictionary to associate each biopsy to the patches and the labels --> d[bio] = {imgs = [list_of_patches_locations]; label = 0...1}
            for i in d['split'][s]:
                img_bio = d['images'][i]['values']['bio']
                img_path = os.path.join(data_root, d['images'][i]['location'])
                if img_bio in self.bios.keys():
                    self.bios[img_bio]['patches'].append(img_path)
                else:
                    img_esrd = d['images'][i]['values']['ESRD']
                    img_fup = float(d['images'][i]['values']['fup'])
                    img_lbl = 0
                    if img_esrd == 'FALSE':
                        img_lbl = 0.5 - min(img_fup, 10) / 20.
                    elif img_fup < 20:
                        img_lbl = 0.5 + (20 - max(img_fup, 10)) / 20.

                    self.bios[img_bio] = {'patches': [img_path], 'label': img_lbl}
            
    def __getitem__(self, index):
        bio = self.bios[list(self.bios.keys())[index]]
        try:
            patches = random.sample(bio['patches'], self.patches_per_bio)
        except ValueError:
            patches = bio['patches']
            patches += [random.choice(bio['patches']) for _ in range(self.patches_per_bio - len(bio['patches']))]
            random.shuffle(patches)

        ground = bio['label']
        images = []

        for patch in patches:
            # image = np.asarray(Image.open(patch))
            image = Image.open(patch)
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        return stack(images), ground, patches

    def __len__(self):
        return len(self.bios.keys())

#DATALOADER

class YAML10YBiosDatasetAllPpb(data.Dataset):
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
               
                

        #DIZIONARIO
        w = csv.writer(open("/homes/grosati/Medical/dizionario_label.csv", "w"))

        # loop over dictionary keys and values
        for key, val in self.bios.items():

            # write every key and value to file
            w.writerow([key, val])
  
    def __getitem__(self, index):

        bio = self.bios[list(self.bios.keys())[index]]
        ppb_wsi = len(bio['images'])
        patches = random.sample(bio['images'], ppb_wsi)
        
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
    
#WSI NORMALI
class YAML10YBiosDataset(data.Dataset):
    # PATCHES mean values BGR = [0.81341412 0.76660304 0.83704776] | std values BGR = [0.14812355 0.18829341 0.12363736]
    # [0.74629832 0.67295842 0.78365591] | std values RGB = [0.17482606 0.21674619 0.14285819]
    #def __init__(self, years, dataset, crop_type, patches_per_bio, transforms=None, split=['training']):
    def __init__(self, dataset, crop_type, patches_per_bio, transforms=None, split=['training']):
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
        return stack(images), ground

    def __len__(self):
        return len(self.bios.keys())


def debug_plot(img, cmap=None):
    from matplotlib import pyplot as plt
    import numpy as np
    img = np.array(img)
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show(block=False)

#FLUO
class YAML10YBiosDatasetFluo(data.Dataset):

    def __init__(self, dataset, crop_type, patches_per_bio, transforms=None, fluo_transform=None, augmentation_transform = None, split=['training','test']):
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
        self.augmentation_transform = augmentation_transform
        self.dataset = dataset
        self.transforms = transforms
        self.fluo_transform = fluo_transform
        self.bios = {}
        counter_miss=0
        counter_found=0
        self.imgs_root = '/nas/softechict-nas-1/fpollastri/data/istologia/images/'
        self.dicts={'0':0,'1':0}
        #DATASET
        #Vecchio file errato con 2k immagini
        #df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/2_Fluo_Id_Bio.xlsx", header = 0)

        #Nuovo file
        df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/HandFLUO-Bio-IMG-Person.xlsx", header = 0)
        #File generale con solo ID_BIOPSIA e ID_IMMAGINE
        #df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/Mtb_d_mdb.xlsx", header=0)
        #Trasformo in int gli ID di Biopsia
        #df_fluo["Biopsia n."]=df_fluo["Biopsia n."].apply(int)

        #all images sara una lista con path img1, path img2...
        all_images = glob.glob(self.imgs_root + '*.tif')

        #Legge il file yaml dove dentro ci sono tutte le info del dataset e lo mette nella variabile d
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        #split train test e eval, prendo tutte le immagini dentro quello split li
        #dicts = {'0':0,'1':0}
    
        for s in split:
            for i in d['split'][s]:
                img_bio = d['bios'][i]['bio']
        
                #PER MTB_D_MDB DEVO USARE IL TIPO STRINGA, PER GLI ALTRI INT
                #id_images = df_fluo[df_fluo['Biopsia n.'] == img_bio]

                id_images = df_fluo[df_fluo['Biopsia n.'] == int(img_bio)]
                id_images = id_images[id_images['Type']=='IgA']['Id_image']
                
                #imgs_path = [img for img in all_images if f'_{img_bio}_pas' in img]

                image_path=[]
                for id in id_images:
                    id_image = '/nas/softechict-nas-1/fpollastri/data/istologia/images/'+ id
                    image_path.append(id_image)
                    counter_found+=1
               
                if image_path == []:
                    print(f'bio {img_bio} has no images')
                    counter_miss+=1
                    continue
                img_esrd = d['bios'][i]['ESRD']
                img_fup = float(d['bios'][i]['fup'])
                img_lbl = 0
                if img_esrd == 'FALSE':
                    img_lbl = 0.5 - min(img_fup, self.years) / (2. * self.years)
                elif img_fup < (2. * self.years):
                    img_lbl = 0.5 + ((2. * self.years) - max(img_fup, self.years)) / (2. * self.years)
        
                if img_lbl > 0.5 :
                    self.dicts['1'] +=len(image_path)
                else:
                    self.dicts['0'] +=len(image_path)
             

                #DISACCOPPIATO
                
                for img in image_path:
                    substring=img[55:]
                    self.bios[substring] = {'images' : img,'label': img_lbl}
                
                #alla fine avro questo dizionario dove dentro avro il nome dell immagine, il path e la label
                #self.bios[img_bio] = {'images': image_path, 'label': img_lbl}
                
        print("Il numero di immagini mancanti è:", counter_miss) 
        print("Il numero di immagini trovate è:", counter_found)   
        print(self.dicts)



        #DIZIONARIO
        w = csv.writer(open("/homes/grosati/Medical/dizionario_fluo_label.csv", "w"))

        # loop over dictionary keys and values
        for key, val in self.bios.items():

            # write every key and value to file
            w.writerow([key, val])


    def get_labels(self):
        return self.dicts
    
    
    #DISACCOPPIO
    def __getitem__(self, index):
    
        fluo_Id = self.bios[list(self.bios.keys())[index]]
        #print("{0} is the fluo_Id".format(fluo_Id))
        fluo_img= fluo_Id['images']
        #print("{0} is the fluo_img".format(fluo_img))
        fluo_label = fluo_Id['label']
        #print("{0} is the fluo_label".format(fluo_label))

        image = Image.open(fluo_img)
             
        if self.transforms is not None:
            
            image = self.transforms(image) 
            #image.show()

            if image.shape == torch.Size([1, 772, 1040]):
                image=torch.cat([image,image,image],0)

            if self.fluo_transform is not None:
                image=image.float()
                image = self.fluo_transform(image)
                
        return image, fluo_label
    
    def __len__(self):
        return len(self.bios.keys())
    
#FLUO NOT DECOUPLED
class YAML10YBiosDatasetFluoNotDecoupled(data.Dataset):

    def __init__(self, dataset, crop_type, patches_per_bio, transforms=None, fluo_transform=None, augmentation_transform = None, split=['training','test']):
 
        
        self.years = 5
        self.patches_per_bio = patches_per_bio
        self.augmentation_transform = augmentation_transform
        self.dataset = dataset
        self.transforms = transforms
        self.fluo_transform = fluo_transform
        self.bios = {}
        counter_miss=0
        counter_found=0
        self.imgs_root = '/nas/softechict-nas-1/fpollastri/data/istologia/images/'
        self.dicts={'0':0,'1':0}
        #DATASET
        #Vecchio file errato con 2k immagini
        #df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/2_Fluo_Id_Bio.xlsx", header = 0)

        #Nuovo file
        df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/HandFLUO-Bio-IMG-Person.xlsx", header = 0)
        #File generale con solo ID_BIOPSIA e ID_IMMAGINE
        #df_fluo = pd.read_excel("/homes/grosati/Medical/data_csv/Mtb_d_mdb.xlsx", header=0)
        #Trasformo in int gli ID di Biopsia
        #df_fluo["Biopsia n."]=df_fluo["Biopsia n."].apply(int)

        #all images sara una lista con path img1, path img2...
        all_images = glob.glob(self.imgs_root + '*.tif')

        #Legge il file yaml dove dentro ci sono tutte le info del dataset e lo mette nella variabile d
        with open(self.dataset, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        #split train test e eval, prendo tutte le immagini dentro quello split li
        #dicts = {'0':0,'1':0}
    
        for s in split:
            for i in d['split'][s]:
                img_bio = d['bios'][i]['bio']
        
                #PER MTB_D_MDB DEVO USARE IL TIPO STRINGA, PER GLI ALTRI INT
                #id_images = df_fluo[df_fluo['Biopsia n.'] == img_bio]

                id_images = df_fluo[df_fluo['Biopsia n.'] == int(img_bio)]
                id_images = id_images[id_images['Type']=='IgA']['Id_image']
                
                #imgs_path = [img for img in all_images if f'_{img_bio}_pas' in img]

                image_path=[]
                for id in id_images:
                    id_image = '/nas/softechict-nas-1/fpollastri/data/istologia/images/'+ id
                    image_path.append(id_image)
                    counter_found+=1
               
                if image_path == []:
                    print(f'bio {img_bio} has no images')
                    counter_miss+=1
                    continue
                img_esrd = d['bios'][i]['ESRD']
                img_fup = float(d['bios'][i]['fup'])
                img_lbl = 0
                if img_esrd == 'FALSE':
                    img_lbl = 0.5 - min(img_fup, self.years) / (2. * self.years)
                elif img_fup < (2. * self.years):
                    img_lbl = 0.5 + ((2. * self.years) - max(img_fup, self.years)) / (2. * self.years)
        
                if img_lbl > 0.5 :
                    self.dicts['1'] +=len(image_path)
                else:
                    self.dicts['0'] +=len(image_path)
             

                #alla fine avro questo dizionario dove dentro avro il nome dell immagine, il path e la label
                self.bios[img_bio] = {'images': image_path, 'label': img_lbl}
                
         
        print("Il numero di immagini mancanti è:", counter_miss) 
        print("Il numero di immagini trovate è:", counter_found)   
        print(self.dicts)



        #DIZIONARIO
        w = csv.writer(open("/homes/grosati/Medical/dizionario_fluo_label.csv", "w"))

        # loop over dictionary keys and values
        for key, val in self.bios.items():

            # write every key and value to file
            w.writerow([key, val])


    def get_labels(self):
        return self.dicts
    
     #RESTITUISCE DATO L'INDICE L'ELEMENTO CORRISPONDENTE
    def __getitem__(self, index):
        #print(index)
        bio = self.bios[list(self.bios.keys())[index]]
        try:
            #OGNI IMMAGINE HA DIVERSE PATCH ASSOCIATE E QUI IO VADO A PRENDERE UN SAMPLE RANDOMICO DI ALCUNE PATCHES
            patches = random.sample(bio['images'], self.patches_per_bio)
        except ValueError:
            patches = bio['images']
            patches += [random.choice(bio['images']) for _ in range(self.patches_per_bio - len(bio['images']))]
            random.shuffle(patches)

        ground = bio['label']
        images = []

        for patch in patches:
         
            image = Image.open(patch)
        
            if self.transforms is not None:
            
               image = self.transforms(image) 
               #image.show()

            #DA USARE PER NORMALIZZAZIONE
            if image.shape == torch.Size([1, 772, 1040]):
                image=torch.cat([image,image,image],0)

                
            if self.fluo_transform is not None:
                image=image.float()
                image = self.fluo_transform(image)
                #check the bit-depth
                
            
            s = image.shape
            #print(s)
           
            images.append(image)

        #Qui ritorna un batch con tutte quelle patch una sopra all'altra
        return stack(images), ground
    
    def __len__(self):
        return len(self.bios.keys())
    




if __name__ == '__main__':

    from torchvision import transforms
    from torch.utils.data import DataLoader

    # preprocess_fn = transforms.Compose([transforms.RandomCrop((1000, 2000), pad_if_needed=True, fill=255)])
    # preprocess_fn = transforms.Compose([transforms.RandomCrop((1000, 2000), pad_if_needed=True, fill=255), transforms.Resize(size=(256, 512))])
    # preprocess_fn = transforms.Resize((256, 256))

    #dname='/nas/softechict-nas-2/fpollastri/data/big_nephro/big_nephro_bios_dataset.yml'
    dname = '/nas/softechict-nas-2/nefrologia/patches_dataset/big_nephro_5Y_bios_dataset.yml'
  
    custom_training_transforms = transforms.Compose([
        
        transforms.ColorJitter(contrast=(1.7, 1.7)),
        transforms.ToTensor(),
        #transforms.Resize((256,256)),
        #PER LE FLUO RIMANE TO_TENSOR E RESIZE
    
    ])
    
    ppb = 1
    dataset = YAML10YBiosDataset(dataset=dname, crop_type='patches', patches_per_bio=ppb,transforms=custom_training_transforms, split=['test'])
    #dataset = YAML10YBiosDatasetFluo(dataset=dname, crop_type='patches', patches_per_bio=ppb,transforms=custom_training_transforms,fluo_transform=None, split=['training'])
    
    
    #FLUO (IMMAGINI) IN TRAIN SONO 146(0) 14 (1) (LE BIO SONO 73 (0) E 7 (1))
    #FLUO (IMMAGINI) IN TEST SONO 65 (0) 29 (1) (LE BIO SONO 27 (0) E 13 (1))
    #WSI (BIO) IN TRAIN SONO 330 (0) 33 (1) 
    #WSI (BIO) IN TEST SONO 88 (0) 45 (1) 
    
    '''
    dict_label=dataset.get_labels()
    size_0 = dict_label.get('0')
    size_1 = dict_label.get('1')
    print(size_0)
    print(size_1)
    '''

    
    tmp = np.array([dataset.bios[k]['label'] for k in dataset.bios])
    tmp[tmp <= 0.5] = 0
    tmp[tmp > 0.5] = 1
    print(tmp.sum())
    print(len(tmp) - tmp.sum())
    

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             # drop_last=True,
                             pin_memory=False)


    rgb = np.zeros((ppb * 450, 3, 256, 256))
    #rgb = np.zeros((ppb * 450, 3, 772, 1040))
    counter = 0

    
    '''#DISACCOPPIAMENTO
    for i, (x,target) in enumerate(data_loader):
        #print(x.shape)
        #print(target)
        if i % 10 == 0:
            print(f'doing batch #{i}')
    '''
    
    '''
        for img in imgs:
            for s_img in img:
                #debug_plot(np.moveaxis(np.array(s_img), 0, -1))
                rgb[counter:counter + ppb] = np.array(s_img)
                counter += ppb
            pass
    '''
    
    #WSI
    for i, (b_img, lbl, name) in enumerate(data_loader):
        #print(b_img, lbl, name)
        if i % 10 == 0:
            print(f'doing batch #{i}')
    
    '''
        for img in b_img:
            for s_img in img:
                #debug_plot(np.moveaxis(np.array(s_img), 0, -1))
                rgb[counter:counter + ppb] = np.array(s_img)
                counter += ppb
            pass
    '''
    
    #print(f'mean values RGB = {np.mean(rgb, axis=(0, 2, 3))} | std values RGB = {np.std(rgb, axis=(0, 2, 3))}')