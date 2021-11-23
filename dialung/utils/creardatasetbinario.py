from torch.utils.data import Dataset
import h5py
import math
from torchvision.transforms import Grayscale
import torch
import numpy as np

class crearDatasetBinarioDIALUNG(Dataset):
    """
    Una clase que hereda los clase torch.utils.data.Dataset para crear un objeto que pueda ser
    manipulado por la clase DataLoader de PyTorch
    ...

    Atributos
    ----------
    img_labels : pandas.DataFrame
        un objeto de tipo pandas.Dataframe que tiene dos columnas, la primera nombre de la imagen y segunda condicion
    img_dir : str
        Directorio de las imagenes de los datasets
    transform : torchvision.transforms
        Transformaciones de formato PyTorch para aplicar las imagenes
    target_transform : int
        Especifica las transformaciones de la etiqueta y características

    Métodos
    -------
    __len__(self)
        Devuelve el tamaño del dataframe
    """
    def __init__(self, ruta_archivo_binario, transform=None, target_transform=None, rgb = False, debug = False):
        """
        Parámetros
        ----------
        ruta_archivo_binario :str
            ruta al archivo binario en formato hdf5
        transform : torchvision.transforms
            Transformaciones de formato PyTorch para aplicar las imagenes
        target_transform : int
            Especifica las transformaciones de la etiqueta y características
        rgb : bool
            Especifica si el modelo de red acepta RGB o solo escala de grises
        debug : bool
            Verifica si hay valores invalidos en la imagen, o si es oslo blanca o negra
        """
        self.ruta_archivo_binario = ruta_archivo_binario
        self.archivo_binario = None
        self.imagenes = None
        self.condiciones = None
        self.transform = transform
        self.target_transform = target_transform
        self.rgb = rgb
        if not self.rgb:
            self.transformacion_gris = Grayscale(num_output_channels=1)
        self.debug = debug
        self.tamano_dataset = 0
        with h5py.File(ruta_archivo_binario, 'r') as archivo:
            name, _ = next(iter(archivo.items()))
            self.tamano_dataset = len(archivo[name])


    def __len__(self):
        """
        Devuelve
        ----------
        int
            El tamaño del Dataset en entero
        """
        return self.tamano_dataset

    def __getitem__(self, idx):
        """Obtiene la imagen del idx del dataset, lee en formato Tensor, aplica las transformaciones
        y devuelve una imagen y su etiqueta.
        Parámetros
        ----------
        idx : int
            índice secuencial de la imagen que se localiza en el DataFrame
        Devuelve
        ----------
        torch.Tensor
            La imagen en formato tensor de PyTorch en Escala de Grises (Tensor[1, alto_imagen, ancho_imagen])
        int
            La etiqueta correspondiente de la imagen
        """
        if self.archivo_binario is None:
            self.archivo_binario = h5py.File(self.ruta_archivo_binario, 'r') #Fuente https://github.com/pytorch/pytorch/issues/11929
            self.imagenes = self.archivo_binario["image"]
            self.condiciones = self.archivo_binario["condition"]

        image = self.imagenes[idx]
        image = torch.from_numpy(image) #convertir a TorchTensor
        if not self.rgb:
            image = self.transformacion_gris(image)

        if self.debug:
            elementos = torch.numel(image)
            unos = (image >= 204).sum()
            zeros = (image <= 51).sum()
            if (unos/elementos > .8) or (zeros/elementos > .8):
                print("Elemento desbalanceado en la carga: {}".format(self.archivo_binario["image_path"][idx]))
        label = self.condiciones[idx].astype(np.int64) #Carga la etiqueta
        image = torch.unsqueeze(image, 3) ## Añade una dimensión al vector 4D
        if self.transform: #Aplico transformacion(es) de la imagen
            image = self.transform(image)
        if self.target_transform: #Aplico transformacion(es) a los labels
            label = self.target_transform(label)
        image = image.squeeze(dim=3) #Volver a formato 3D
        #Verificar por valores ilegales en los datos como nan o inf producto de la transformacion
        if self.debug:
            if torch.isnan(image).any() or torch.isinf(image).any():
                print("La imagen: {} contiene valores no válidos".format(self.archivo_binario["image_path"][idx]))
            if math.isnan(label) or math.isinf(label):
                print("No existe el label de la imagen: {}".format(self.archivo_binario["image_path"][idx]))
            elementos = torch.numel(image)
            unos = (image >= .8).sum()
            zeros = (image <= .2).sum()
            if (unos/elementos > .8) or (zeros/elementos > .8):
                print("Elemento desbalanceado después de la transformación: {}".format(self.archivo_binario["image_path"][idx]))
        return image, label
