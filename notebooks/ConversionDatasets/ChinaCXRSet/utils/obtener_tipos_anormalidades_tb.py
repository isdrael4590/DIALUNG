 #!/usr/bin/python3
# -*- coding: utf-8 -*-
#Program by Jose Jacome josejacomeb@gmail.com
'''
Programa para obtener las anormalidades del Dataset de Shenzhen chest X-ray set
'''

import os
import argparse

ruta_clinical_readings="ClinicalReadings/"
anormalidades_dataset = []

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Comandos para ejectuar el programa para obtener las anormalidad del Dataset ChinaCXRSet")
    # -- Create the descriptions for the commands
    d_desc = "La ruta al dataset descargado ChinaCXRSet"

    # -- Create the arguments
    parser.add_argument("-d", "--dataset", help=d_desc, required=True)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    ruta_completa_clinical_readings = os.path.join(args.dataset, ruta_clinical_readings)
    nombres_archivos_cr = os.listdir(ruta_completa_clinical_readings)

    for archivo_cr in nombres_archivos_cr:
        ruta_completa_archivo_cr = os.path.join(ruta_completa_clinical_readings, archivo_cr)
        with open(ruta_completa_archivo_cr, 'r') as archivo_texto:
            lineas = archivo_texto.read()
            anormalidades_archivo = lineas.split('\n')[-1].split('\t') # Remueve el enter y la tabulación
            for anormalidad in anormalidades_archivo:
                anormalidad = anormalidad.lstrip()
                anormalidad = anormalidad.rstrip()
                if anormalidad and anormalidad not in anormalidades_dataset:
                    anormalidades_dataset.append(anormalidad)
    print("Anormalidades detectadas en el dataset {}".format(args.dataset))
    for anormalidad in anormalidades_dataset:
        print("- {}".format(anormalidad))

if __name__ == '__main__':
    main()
