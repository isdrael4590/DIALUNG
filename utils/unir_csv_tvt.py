#!/usr/bin/python3
# -*- coding: utf-8 -*-
#Program by Jose Jacome josejacomeb@gmail.com
'''
Programa unir los datasets únicos de formato DIALUNG hacia los tres necesarios para el entrenamiento
'''

import argparse
import os
import pandas as pd

def get_args():
    '''
        Obtener los argumentos de la línea de comandos
    '''
    parser = argparse.ArgumentParser("Comandos para ejectuar el programa para reunir los csvs de los datasets en formato DIALUNG")
    # -- Create the descriptions for the commands
    d_desc = "La ruta al dataset descargado ChinaCXRSet"

    # -- Create the arguments
    parser.add_argument("-d", "--directorio", help=d_desc, required=True)

    args = parser.parse_args()
    return args

def df_archivos_unidos(directorio, array_nombres_archivo):
    df_archivo = pd.DataFrame()
    for nombre_archivo in array_nombres_archivo:
        ruta_csv = os.path.join(directorio, nombre_archivo)
        df_csv = pd.read_csv(ruta_csv)
        nombre_columnas = list(df_csv.columns)
        if len(nombre_columnas) > 2:
            df_csv.drop([df_csv.columns[0]], axis = 1, inplace = True) #Remover indice
        nombre_columnas = list(df_csv.columns)
        if nombre_columnas[0] != "image_name" or nombre_columnas[1] != "condition":
            print({nombre_columnas[0]: "image_name", nombre_columnas[1]: "condition"})
            df_csv.rename(columns={nombre_columnas[0]: "image_name", nombre_columnas[1]: "condition"}, inplace = True) #nombre columnas
        df_archivo = pd.concat([df_archivo, df_csv])

    return df_archivo

def main():
    args = get_args()
    archivos_carpeta_csv = os.listdir(args.directorio)
    dataset_train = [archivo for archivo in archivos_carpeta_csv if "train" in archivo]
    dataset_test = [archivo for archivo in archivos_carpeta_csv if "test" in archivo]
    dataset_val = [archivo for archivo in archivos_carpeta_csv if "val" in archivo]

    df_train = df_archivos_unidos(args.directorio, dataset_train)
    df_test = df_archivos_unidos(args.directorio, dataset_test)
    df_val = df_archivos_unidos(args.directorio, dataset_val)

    df_train.to_csv("train.csv")
    df_test.to_csv("test.csv")
    df_val.to_csv("val.csv")

if __name__ == '__main__':
    main()
