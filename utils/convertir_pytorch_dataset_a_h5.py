#!/usr/bin/python3
# -*- coding: utf-8 -*-
#Program by Jose Jacome josejacomeb@gmail.com
'''
Programa para convertir un dataset de tipo csv y carpetas a un solo
archivo en formato binario de tipo h5
'''
import h5py
import numpy as np
import pandas as pd
import cv2
import os
import torch
import argparse
from tqdm import tqdm

def get_args():
    '''
        Obtener los argumentos de la línea de comandos
    '''
    parser = argparse.ArgumentParser("Comandos para convertir el dataset DIALUNG a formato Binario")
    # -- Create the descriptions for the commands
    c_desc = "Ruta al archivo csv a ser convertido"
    a_desc = "Nombre del archivo de salida "
    d_desc = "Directorio base donde se encuentran las imagenes"
    r_desc = "Resolucion de las imagenes"
    # -- Create the arguments
    parser.add_argument("-c", "--csv", help=c_desc, required=True)
    parser.add_argument("-a", "--archivo", help=a_desc, required=True)
    parser.add_argument("-d", "--directorio", help=d_desc, required=True)
    parser.add_argument("-r", "--resolucion", help=r_desc, default=256, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    nombre_archivo = args.archivo
    if not ".hdf5" in nombre_archivo:
        nombre_archivo += ".hdf5"
    dataset_pd = pd.read_csv(args.csv, index_col = 0) #Leo el archivo CSV
    nombre_dataset = os.path.basename(args.csv).split(".")[0]
    images = []
    labels = []
    images_path = []
    tamano_transformacion = None
    if type(args.resolucion) == int:
        tamano_transformacion = (args.resolucion, args.resolucion)
    elif type(args.resolucion) == tuple:
        tamano_transformacion = args.resolucion
    tamano_dataset = len(dataset_pd)
    for index, row in tqdm(dataset_pd.iterrows(), total=tamano_dataset):
        ruta_imagen = os.path.join(args.directorio, row["image_name"])
        image = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convertir a RGB
        image = cv2.resize(image, tamano_transformacion)
        image = np.transpose(image, (2, 0, 1)) #Convertir de HxWxC a CxHxW
        images.append(image)
        labels.append(int(row["condition"]))
        images_path.append(row["image_name"])
    images = np.array(images, dtype="uint8")
    labels = np.array(labels, dtype="int32")
    print("Tamaño del dataset imagenes: {}".format(images.shape))
    print("Tamaño del dataset labels: {}".format(labels.shape))

    with h5py.File(nombre_archivo, "w") as f:
        dt = h5py.string_dtype(encoding='utf-8')
        images_path = np.array(images_path, dtype=dt)
        images_h5set = f.create_dataset("image", images.shape, dtype="uint8", data=images)
        images_h5set = f.create_dataset("condition", labels.shape, dtype='int32', data=labels)
        images_h5set = f.create_dataset("image_path", data = images_path)

if __name__ == "__main__":
    main()
