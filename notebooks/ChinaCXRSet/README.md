 
# China Set - The Shenzhen set - Chest X-ray Database

Dataset obtenido de la siguiente dirección: [ChinaCXRSet Link](http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip)

## Características imágenes
### Formato 
PNG
### Resolución
Aproximadamente 3K x 3K píxeles

## Distribución del dataset
- 336 casos de pacientes con manifestación de tuberculosis
- 326 casos normales


## Atributos
- Género
- Edad
- Indica la anormalidad(Sano o Enfermo con Manifestación de Tuberculosis)
### Tipos de anormalidad en el dataset
- normal
- normale
- STB
- NATB
- TB
- tuberculosis pleuritis
- ATB
- PTB in the left lower field
- Bilateral secondary PTB , right pleural change after decortication
- bilateral PTB ?right pleurisy with pleural effusion
- PTB in the left upper field
- secondary PTB in the right upper field
- secondary PTB in the bilateral upper field
- PTB in the right upper field
- bilateral PTB
- secondary PTB  in the right upper field
- bilateral PTB,right upper field atelectasis
- Bilateral secondary PTB
- Bilateral secondary PTB, left encapsulated intrathoracic fluid
- PTB in the right lower field
- Right PTB
- Left PTB, left pleural thickening
- PTB in the bilateral upper and middle field
- Old PTB in the right upper field
- left PTB,pleural thickening
- bilateral PTB, with left pleural effusion
- PTB  in the right upper field
- left PTB
- PTB in the bilateral upper field
- left PTB,left pleurisy
- secondary PTB  in the bilateral upper field
- Right pneumothorax, Right upper PTB, bilateral widespread infection
- right upper PTB, pleural adhesions in  left lower field
- right upper PTB
- bilateral PTB, right pleurisy
- right upper PTB with fibrous changes
- Right PTB with fibrous changes
- bilateral PTB,  right upper lobe atelectasis
- secondary PTB  in the bilateral upper fields
- left secondary PTB
- PTB in the right upper field,COPD
- secondary PTB  in the left upper field
- right secondary PTB, right pleurisy with  pleural effusion
- left upper PTB, left pleurisy
- right PTB,right pneumothorax
- right secondary PTB
- right upper field PTB
- right secondary PTB  with bilateral  pleurisy
- bilateral secondary PTB
- PTB in the bilateral upper fields, left pleurisy
- secondary PTB  in the right  middle field
- Left PTB
- PTB in the middle lower field
- PTB in the bilateral upper fields
- PTB
- bilateral hematogenous disseminated PTB
- right upper pneumonia
- bilateral PTB, with cavity formation in right upper field
- left secondary PTB,right pleural effusion
- right lower field PTB
- bilateral subacute hematogenous disseminated PTB
- Bilateral secondary PTB with left lower field pneumonia
- bilateral acute hematogenous disseminated PTB
- Bilateral secondary PTB  with most lesions calcified
- 1.secondary PTB  in the right upper field; 2.small tuberculoma; 3.right pleural thickening and adhesions
- Left secondary PTB in the upper and middle fields, mainly  fibrous  lesions
- secondary PTB  in the left lower field
- secondary PTB in the bilateral upper and middle fields, mainly fibrous  lesions
- bilateral secondary PTB with multiple cavitations
- bilateral secondary PTB,mainly hyperplastic lesions
- Right secondary PTB,mainly hyperplastic lesions
- secondary PTB  in thel left upper field
- bilateral secondary PTB  with pleural thickening
- Right secondary PTB
- secondary PTB in the right upper field,mainly hyperplastic lesions
- Right secondary PTB in the upper and middle fields
- bilateral secondary PTB  with right pneumothorax
- 1.bilateral secondary PTB  with right upper atelectasis;2.right pleural adhesions;3.left compensatory emphysema
- bilateral secondary PTB  with right pleural thickening


## Formato  de Dataset
Propio, el dataset contiene dos Carpetas **CXR_png/** y **ClinicalReadings/**, estás son descritas a continuación
### CXR_png/
Las imágenes contienen un nombre con el formato *CHNCXR_XXXX_0.png*, el nombre está dividido por el guión bajo por tres partes, la primera parte es el nombre del Dataset, la segunda especifica el número de imágen y la tercera si el valor es '0' significa que es un CXR Normal, en el caso de que sea '1' la imágen  corresponde a un pulmón anormal.
### ClinicalReadings/
En ésta carpeta se encuentran con el mismo formato de nombre, los atributos del paciente en formato txt, ej: *CHNCXR_XXXX_0.txt*, en la primera fila del archivo se encuentran los atributos *género* y *edad* separados por un espacio simple. En la(s) siguiente fila(s), se especifíca la enfermad 

## Citación

- 1) Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue Z, Palaniappan K, Singh RK, Antani S, Thoma G, Wang YX, Lu PX, McDonald CJ.  Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 2014 Feb;33(2):233-45. doi: 10.1109/TMI.2013.2284099. PMID: 24108713
- 2) Candemir S, Jaeger S, Palaniappan K, Musco JP, Singh RK, Xue Z, Karargyris A, Antani S, Thoma G, McDonald CJ. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014 Feb;33(2):577-90. doi: 10.1109/TMI.2013.2290491. PMID: 24239990