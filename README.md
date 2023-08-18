# Medical
* union_feature extraction: estrae le features da wsi e fluo post aggregazione(features_train, features_test)
* linear_classifier: (tentativo di allenare un classificatore lineare sulle features estratte in precedenza)
* eval_fluo_wsi: evaluation su wsi e fluo, il dataset consiste nell'intersezione delle wsi con le fluo, l'output finale è la media dell'output ottenuto dalla valutazione delle 2 reti sulle wsi separate dalle fluo della stessa biopsia.
* main_wsi: train ed eval del modello usando le wsi
* main_fluo: train ed eval del modello usando le fluo
* big_nephro_dataset: insieme di tutti i dataloader
