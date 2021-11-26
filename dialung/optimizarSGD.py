#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Program by Jose Jacome josejacomeb@gmail.com

"""
    Programa para tuner los hiperparámetros del Optimizador SGD
"""
import numpy as np
import torch
from ray import tune
from torch.optim.lr_scheduler import OneCycleLR
from torch import nn


class OptimizarSGDTrainable(tune.Trainable):
    def setup(self, config, dataloader, modelo_func, dispositivo, epocas=10):
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.nesterov = config["nesterov"]
        self.numero_canales_entrada = config["numero_canales_entrada"]
        self.clases_salida = config["clases_salida"]
        self.dataloader = dataloader
        self.modelo_func = modelo_func
        self.dispositivo = dispositivo
        self.epocas = epocas
        self.modelo = None
        self.scheduler = None
        self.optimizador = None

    def step(self):  # This is called iteratively.
        pesos_clase_desbalanceada = torch.tensor(
            [0.5097178, 2.12377522, 1.0953478, 1.52830776]
        ).to(self.dispositivo)
        self.funcion_perdida = nn.CrossEntropyLoss(
            weight=pesos_clase_desbalanceada
        )  # Uso de CrossEntropy
        self.modelo = self.modelo_func(
            numero_canales_entrada=self.numero_canales_entrada,
            clases_salida=self.clases_salida,
        ).to(self.dispositivo)
        self.optimizador = torch.optim.SGD(
            self.modelo.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
        self.scheduler = OneCycleLR(
            self.optimizador,
            max_lr=self.lr,
            steps_per_epoch=len(self.dataloader),
            epochs=self.epocas,
        )
        self.modelo.train()  # Poner el modelo en modo de entrenamiento
        totalTrainLoss = 0  # Variable para guardar los datos del loss por iteración
        trainCorrect = 0  # Variable para almacenar las clasificaciones correctas
        print(self.training_iteration)
        iteraciones_dataloader = len(self.dataloader)
        total_elementos_dataloader = len(self.dataloader.dataset)
        # Detectar anomalias with torch.autograd.detect_anomaly():
        for step in range(self.epocas):
            for batch, (X, y) in enumerate(self.dataloader):
                X = X.to(self.dispositivo)
                y = y.to(self.dispositivo)
                # Backpropagation
                self.optimizador.zero_grad()
                # Computar la predicción y el loss
                pred = self.modelo(X)
                loss = self.funcion_perdida(pred, y)
                loss.backward()
                # nn.utils.clip_grad_value_(self.modelo.parameters(), clip_value=1.0) #Clipping gradients
                self.optimizador.step()
                self.scheduler.step()
                totalTrainLoss += loss.item()
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            avgTrainLoss = totalTrainLoss / iteraciones_dataloader
            trainCorrect = trainCorrect / total_elementos_dataloader
        return {"loss": avgTrainLoss, "accuracy": trainCorrect, "done": True}
