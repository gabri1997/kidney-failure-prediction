# Prediction of kidney failure using Neural Networks

The purpose of this study is to investigate and enhance the prognostic score of the learning algorithm (DLPS). By combining the results obtained using whole slide images of renal biopsies (WSI) with their counterpart in immunofluorescence, it becomes possible to predict whether patients will experience renal failure within the next 5 years.

The experiments concern three different types of images:
- Whole slide images (WSI)
- Immunofluorescence images (Fluo)
- Combination of the two aforementioned types (WSI + Fluo)

<div>
    <img src="Imges/id1006_0071_pas_Regione 1_1CC0_patch_2.png" alt="Alt text" title="Patch of a wsi">
    <img src="Imges/00045.png" alt="Alt text" title="IgA immunofluorescne image">
</div>



## Whole slide images (WSI)

The `main_wsi.py` file allows you to run experiments on the WSI (Whole Slide Images).

There are several input parameters, the most important of which is as follows:
* `type_of_experiment`: Specifies how WSI patches are loaded. This parameter can be either "standard" or "all_patches".
  In the "standard" case, a fixed number of patches are considered for each WSI, specified by a second parameter (`patches_per_bio`). If a WSI
  has fewer patches than indicated by the parameter, the missing patches will be randomly re-extracted from the previous ones.
  The second type, "all_patches", considers all patches that make up each WSI.
* `patches_per_bio`: Specifies the number of patches to be extracted from each WSI. This value is the same for each WSI and is used only when
  the `type_of_experiment` parameter is set to "standard".

The other parameters are typical (number of epochs, preprocessing type, model type, etc.).
To execute the `main_wsi.py` file, it's necessary to have the `big_nephro_dataset` file, which enables image loading.
The class responsible for loading images in the `big_nephro_dataset` file is:
* `YAML10YBiosDataset`: Used in the training.
* `YAML10YBiosDataset` or `YAML10YBiosDatasetAllPpb`: Used in testing, depending on the type of experiment (`YAML10YBiosDatasetAllPpb` for "all_patches").

## Standard version

<img src="Imges/Metrics_rename/WSI standard.png" alt="Alt text" title="Confusion matrix standard WSI">

## AUC

### ResNet-18

<img src="Imges/Metrics_rename/WSI standard(1).png" alt="Alt text" title="AUC">

## All patches per biopsy

<img src="Imges/Metrics_rename/All patches.png" alt="Alt text" title="Confusion matrix all patches">

### ResNet-34

<img src="Imges/Metrics_rename/All patches ResNet-34.png" alt="Alt text" title="Confusion matrix all patches ResNet-34">

## AUC
<img src="Imges/Metrics_rename/WSI ALL PPB.png" alt="Alt text" title="Confusion matrix all patches">

### ResNet-34

<img src="Imges/Metrics_rename/All patches ResNet-34(1).png" alt="Alt text" title="AUC all patches ResNet-34">

## Immunofluorescence images

The `main_fluo.py` file allows you to run experiments on immunofluorescence images.

Immunofluorescence images are not patches; therefore, they are considered individually and inherit the label from the biopsy they originate from.
The input parameters are standard.
The class responsible for loading images in the `big_nephro_dataset` file is:
* `YAML10YBiosDatasetFluo`: Used for both training and testing.

## Fluo unweighted

<img src="Imges/Metrics_rename/Fluo no weights.png" alt="Alt text" title="Confusion matrix fluo unweighted">

## AUC

<img src="Imges/Metrics_rename/Fluo no weights(1).png" alt="Alt text" title="AUC fluo unweighted">

## Fluo weighted and decoupled

<img src="Imges/Metrics_rename/Fluo decoupled.png" alt="Alt text" title="Confusion matrix fluo weighted">

## AUC

<img src="Imges/Metrics_rename/Fluo decoupled(1).png" alt="Alt text" title="AUC fluo weighted">

### Comparison

<img src="Imges/Metrics_rename/Fluo no weights vs weights.png" alt="Alt text" title="AUC fluo weighted">

## WSI + Fluo

The `eval_fluo_wsi.py` file allows for evaluation using both types of images.

In the evaluation, immunofluorescence images and the WSI from the same biopsy are considered. The training is carried out separately, where one network is trained using WSI, and another network is trained using fluorescence images.

Subsequently, the weights are loaded and used for evaluation on the intersection of the two datasets. During the evaluation, the two types of images (coming from the same biopsy, one being fluorescence and the other PAS) are classified by the two networks. The final output of the classification is the average of the two values predicted by the networks.

The preloaded weights of the two networks are in the "weights" folder.

<img src="Imges/combined out.png" alt="Alt text" title="Combined output">


## Results
Acc: 0.825 | AUC: 0.877 | F1 Score: 0.533 | Precision: 0.800 | Recall: 0.400 | Specificity: 0.967 | Trues: 5 | Correct Trues: 4 | Ground Truth Trues: 10

## Feature extraction

The `union_feature_extraction.py` file allows extracting features from both WSI patches and immunofluorescence images.

The features are then saved into:
* `features_train.pt`
* `features_test.pt`

## Linear classifier

The `linear_classifier.py` file allows retraining a linear classifier using the previously extracted features.

The classifier is trained using features extracted from both WSI and Fluo images belonging to the same biopsy (`fetaures_train.py`).

Once trained, an evaluation is performed using the features present in the file `features_test.py`.

## Evaluation across various dataset partitions.

One of the experiments conducted in this study involves comparing results across multiple instances of the dataset, referred to as 'bags', that correspond to smaller subset of the dataset itself.
The `wsi_split.py` file enables conducting experiments with multiple splits on the WSI dataset.
The file `filter.py` must be included to create the splits.
Firstly, the traditional division of the dataset into separate training and test splits have been eliminated, instead, the entire dataset was used as a whole.        
The approach involved conducting multiple runs, each involving the selection of a distinct 'bag' from the dataset. To facilitate this, two separate lists were created: one containing indices corresponding to images in the training set, and the other containing indices of images in the test set. Both lists are composed by indices linked to images with labels 0 and 1. The composition of these lists involved randomly selecting a subset of indices from the complete index list, while maintaining a consistent ratio between labels 0 and 1.

Five complete runs were performed, each involving the selection of a distinct bag chosen at random. 
The objective is to compare the outcomes and evaluate the extent to which they are impacted by the specific training and test instances encountered by the model.

These experiments were conducted for both Whole Slide Images and Immunofluorescence images.

Specifically, the main change that has been done consists in the use of the `SubsetRandomSampler`
that is a class provided by the PyTorch library, specifically in the `torch.utils.data module`.

The dataset used is in the folder Datasets:  `big_nephro_5Y_bios_dataset_split.yml`.




