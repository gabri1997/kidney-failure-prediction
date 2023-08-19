# Predition of kidney failure using Neural Networks

Lo scopo di questo studio`e quello di indagare e migliorare lo score prognostico dell’algoritmo di apprendimento (DLPS). Combinando i risultati ottenuti utilizzando immagini intere di biopsie renali (WSI) con la controparte in immunofluorescenza, si riesce a prevedere se i pazienti saranno colpiti da insufficienza renale entro i prossimi 5 anni

Gli esperimenti riguardano 3 diverse tipologie di immagini: 
- Whole slide images (WSI)
- Immagini in immunofluorescenza (Fluo)
- Unione delle due precedenti tipologie (WSI + Fluo)

## Whole slide images

Il file main_wsi.py consente di eseguire gli esperimenti sulle WSI.

I parametri in input sono diversi, il più importante è il seguente:
* type_of_experiment: consente di specifiare il modo in cui vengono caricati i patches delle WSI, questo parametro può essere "standard" o "all_patches".
  Nel caso standard per ogni WSI il numero di patches considerati è fisso ed è specificato da un secondo parametro (patches_per_bio). Nel caso in cui una WSI
  avesse meno patches di quelli indicati dal parametro, i patches mancanti sarebbero riestratti casualmente dai precedenti.
  Il secondo tipo, "all_patches", consente di considerare per ogni WSI, tutti i patches di cui è composta. 
* patches_per_bio: consente di specificare il numero di patch da estrarre da ogni WSI, questo valore è uguale per ciascuna WSI ed è utilizzato solo nel caso in cui
  il parametro type_of_experiment = "standard".
  
Gli altri parametri sono i classici (numero di epoche, tipo di preprocessing, tipo di modello etc.)
Per eseguire il file main_wsi è necessario avere il file big_nephro_dataset che consente di caricare le immagini.
La classe che carica le immagini nel file big_nephro_dataset è:
*YAML10YBiosDataset : nel train
*YAML10YBiosDataset o YAML10YBiosDatasetAllPpb: nel test, a seconda della tipologia di esperimento (YAML10YBiosDatasetAllPpb per "all_patches")

## Immagini in immunogluorescenza

Il file main_fluo.py consente di eseguire gli esperimenti sulle immagini in immunofluorescenza.

Le immagini in immunofluorescenza non sono patches, perciò vengono considerate singolarmente ed ereditano la label della biopsia da cui derivano.
I parametri in input sono classici.
La classe che carica le immagini nel file big_nephro_dataset è:
*YAML10YBiosDatasetFluo : per train e test

## WSI + Fluo

Il file eval_fluo_wsi.py consente di fare una evaluation utilizzando entrambe le tipologie di immagini.
Nell'evaluation vengono considerate le immagini in immunofluorescenza e le WSI che appartenenti alla stessa biopsia.
Il train viene eseguito separatamente, una rete viene addestrata usando le WSI, una seconda rete viene addestrata usando le fluo.
Succcessivamente, i pesi vengono caricati ed utilizzati per fare l'evaluation sulla interesezione dei due dataset.
Nell'evaluation le due tipologie di immagini (provenienti dalla stessa biopsia, ma una fluo e l'altra PAS) vengono fatte classificare dalle due reti, l'output finale della classificazione è la media dei 2 valori predetti dalle reti.
I pesi da precaricare delle due reti sono nella cartella weights.

## Feature extraction

Il file union_feature_extraction.py consente di estrarre le features dai patches delle WSI e dalle immagini in immunofluorescenza.
Le features vengono poi salvate in:
-features_train.pt
-features_test.pt

## Linear classifier

Il file linear_classifier.py permette di riaddestrare un classificatore lineare usando le features precedentemente estratte. 
Il classificatore viene addestrato usando le features estratte da immagini WSI e Fluo che appartengono alla stessa biopsia (fetaures_train.py).
Una volta addestrato, viene fatta l'evaluation con le features presenti nel file features_test.py.


