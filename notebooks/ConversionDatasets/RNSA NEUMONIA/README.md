# NIH Chest X-ray Dataset of 8 Common Thorax Disease

Dataset obtenido de la siguiente dirección: [ChestX-ray8 Link](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## Características imágenes
### Formato 
PNG
### Resolución
Aproximadamente 1024 x 1024 píxeles

### Proyecciones
- Anterior-Posterior(AP)
- Posterior-Anterior(PA)

## Distribución del dataset
- No Finding                      60361
- Infiltration                     9547
- Atelectasis                      4215
- Effusion                         3955
- Nodule                           2705
- Pneumothorax                     2194
- Mass                             2139
- Effusion|Infiltration            1603
- Atelectasis|Infiltration         1350
- Consolidation                    1310
- Atelectasis|Effusion             1165
- Pleural_Thickening               1126
- Cardiomegaly                     1093
- Emphysema                         892
- Infiltration|Nodule               829
- Atelectasis|Effusion|Infiltration 737
- Fibrosis                          727
- Edema                             628
- Cardiomegaly|Effusion             484
- Consolidation|Infiltration        441
- Infiltration|Mass                 420
- Effusion|Pneumothorax             403
- Effusion|Mass                     402
- Atelectasis|Consolidation         398
- Mass|Nodule                       394
- Edema|Infiltration                392
- Infiltration|Pneumothorax         345
- Consolidation|Effusion            337
- Emphysema|Pneumothorax            337
- Pneumonia                         322

## Atributos
- Índice Imagen
- Follow-up
- ID paciente
- Edad
- Proyección(AP y PA)
- Imagen Ancho y Alto
- Espaciado de Imágen en *x* y *y*
### Tipos de anormalidad en el dataset
- Atelectasis 
- Cardiomegaly
- Effusion
- Infiltration
- Mass 
- Nodule 
- Pneumonia 
- Pneumothorax 
- Consolidation 
- Edema 
- Emphysema 
- Fibrosis 
- Pleural_Thickening
- Hernia

## Formato  de Dataset
Propio, el dataset contiene una carpeta **images/** donde se encuertran alojados los 45.7Gb de archivos, los datos están en CSV como por ejemplo los archivos **Data_Entry_2017_v2020** y **BBox_List_2017**

## Citación

- Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017
