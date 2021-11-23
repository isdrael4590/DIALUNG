import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm.notebook as tq
try:
    from backpack import extend
    from cockpit import Cockpit, CockpitPlotter
    from cockpit.utils.configuration import configuration
except ModuleNotFoundError or ImportError as ee:
    print("Modulo no encontrado, por favor instale cockpit-for-pytorch antes de intentar de nuevo: {}".format(ee.msg))

def cargar_librerias_debug():
    #Herramientas Debugger
    exito = True
    print(exito)
    return exito


mapa_condiciones = {
    0: "normal",
    1: "tuberculosis",
    2: "neumonia",
    3: "covid-19"
}

class EntrenamientoValidacionDIALUNG:
    def __init__(self, modelo, funcion_perdida, dispositivo,
        optimizador, scheduler, ruta_writer = None, debugger = False):
        self.entrenamiento_global_step = 0
        # Inicializar un diccionario para la historia del entrenamiento
        self.H = {
          "train_loss": [],
          "train_acc": [],
          "val_loss": [],
          "val_acc": []
        }
        self.epocas = len(self.H)
        self.writer = None #No inicializar por defecto
        self.modelo = modelo #Modelo de PyTorch
        self.funcion_perdida = funcion_perdida #Función perdida asignada
        self.dispositivo = dispositivo #Dispositivo donde ejecutar el entrenamiento
        self.optimizador = optimizador #Optimizador del modelo
        self.scheduler = scheduler #Planificador para reducir el learning rate
        self.mejor_precision = 0
        self.reporte_classificacion = ""
        if ruta_writer:
            self.writer = SummaryWriter(ruta_writer)
        self.debugger = debugger
        self.perdida_individual = None
        if self.debugger:
            try:
                self.modelo = extend(self.modelo, use_converter = True)
            except:
                print("Modelo no soportado por el Debugger, verifique que sea Secuencial o ResNET")
                self.debugger = False
            if self.debugger:
                self.funcion_perdida = extend(self.funcion_perdida)
                self.perdida_individual = torch.nn.CrossEntropyLoss(reduction="none")
                #Parametros de Cockpit
                self.cockpit = Cockpit(self.modelo.parameters(), quantities=configuration("full"))
                self.plotter = CockpitPlotter()

    def entrenar(self, dataloader):
        print("## Entrenamiento ##")
        size = len(dataloader.dataset)
        self.modelo.train() #Poner el modelo en modo de entrenamiento
        tiempo_inicio = time.time() #Calculo de tiempo de inicio
        totalTrainLoss = 0 #Variable para guardar los datos del loss por iteración
        trainCorrect = 0 #Variable para almacenar las clasificaciones correctas
        batch_size = dataloader.batch_size
        train_steps = size // dataloader.batch_size
        num_batches = len(dataloader)
        #Detectar anomalias with torch.autograd.detect_anomaly():
        for batch, (X, y) in tq.tqdm(enumerate(dataloader), total = num_batches, desc = "Progreso dataset entrenamiento(batches)"):
            X = X.to(self.dispositivo)
            y = y.to(self.dispositivo)
            # Backpropagation
            self.optimizador.zero_grad()
            # Computar la predicción y el loss
            pred = self.modelo(X)
            loss = self.funcion_perdida(pred, y)
            if self.debugger:
                losses = self.perdida_individual(pred, y)
                    # backward pass
                info_dict = {
                    "batch_size": batch_size,
                    "individual_losses": losses,
                    "loss": loss,
                    "optimizer": self.optimizador
                }
                with self.cockpit(self.entrenamiento_global_step, info=info_dict):
                    loss.backward(create_graph=self.cockpit.create_graph(self.entrenamiento_global_step))
            else:
                loss.backward()
            #nn.utils.clip_grad_value_(self.modelo.parameters(), clip_value=1.0) #Clipping gradients
            self.optimizador.step()
            if self.scheduler:
                self.scheduler.step()
            self.entrenamiento_global_step += 1
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        #Calculo tiempo final
        tiempo_entrenamiento_epoch = time.time() - tiempo_inicio
        tiempo_entrenamiento_epoch = str(datetime.timedelta(seconds=tiempo_entrenamiento_epoch))
        avgTrainLoss = totalTrainLoss / train_steps
        trainCorrect = trainCorrect / size
        self.H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        self.H["train_acc"].append(trainCorrect)
        if self.writer:
            self.writer.add_scalar('Loss/train', self.H["train_loss"][-1], self.entrenamiento_global_step)
            self.writer.add_scalar('Accuracy/train', self.H["train_acc"][-1], self.entrenamiento_global_step)
            self.writer.add_scalar('Learning rate/lr', self.optimizador.param_groups[0]['lr'], self.entrenamiento_global_step)
        print("Valor Función pérdida: {:.3f} Precisión: {:.2f}%  Tasa de aprendizaje: {}".format(self.H["train_loss"][-1],
                                                                                                self.H["train_acc"][-1]*100,
                                                                                                self.optimizador.param_groups[0]['lr'] ))
        print("Tiempo de iteración: {}".format(tiempo_entrenamiento_epoch))

    def validar(self, dataloader):
        print("## Validación ##")
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size
        test_loss, correct = 0, 0
        tiempo_inicio = time.time() #Calculo de tiempo de inicio
        preds = [] #Guardar las predicciones
        gt = [] #Guardar los valores reales
        with torch.no_grad():
            self.modelo.eval() #Poner en modelo de evaluacion
            for X, y in tq.tqdm(dataloader,  total = num_batches, desc = "Progreso dataset evaluacion (Batches)"):
                X = X.to(self.dispositivo)
                gt.extend(y)
                y = y.to(self.dispositivo)
                pred = self.modelo(X)
                preds.extend(pred.argmax(1).cpu().numpy())
                test_loss += self.funcion_perdida(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        #Calculo tiempo final
        tiempo_val_epoch = time.time() - tiempo_inicio
        tiempo_val_epoch = str(datetime.timedelta(seconds=tiempo_val_epoch))
        test_loss /= num_batches
        correct /= size
        self.H["val_loss"].append(test_loss)
        self.H["val_acc"].append(correct)
        if self.writer:
            self.writer.add_scalar('Loss/val', self.H["val_loss"][-1], self.entrenamiento_global_step)
            self.writer.add_scalar('Accuracy/val', self.H["val_acc"][-1], self.entrenamiento_global_step)
        self.reporte_classificacion = classification_report(gt, preds, target_names=mapa_condiciones.values(), zero_division = 0)
        print(self.reporte_classificacion)
        print("Valor Función pérdida: {:.3f} Precisión: {:.2f} %".format(test_loss, 100*correct))
        print("Tiempo de iteración: {}".format(tiempo_val_epoch))


    def guardar_modelo(self, ruta_modelo, ruta_metricas, epoch):
        self.epocas = epoch
        #Rutas modelos
        ruta_guardar_mejor = os.path.join(ruta_modelo, "mejor.pt")
        ruta_guardar_ultimo = os.path.join(ruta_modelo, "ultimo.pt")
        #Rutas metricas
        ruta_reporte_clasificacion = os.path.join(ruta_metricas, "ultimo_reporte.txt")
        ruta_mejor_reporte_clasificacion = os.path.join(ruta_metricas, "mejor_reporte.txt")
        ruta_imagen_precision = os.path.join(ruta_metricas, "ultima_precision.png")
        ruta_imagen_perdida = os.path.join(ruta_metricas, "ultima_perdida.png")
        ruta_imagen_mejor_precision = os.path.join(ruta_metricas, "mejor_precision.png")
        ruta_imagen_mejor_perdida = os.path.join(ruta_metricas, "mejor_perdida.png")

        plt.style.use("ggplot")
        plt.ioff()
        plt.figure(1) #Figura precision
        plt.plot(self.H["train_acc"], label = "train_acc")
        plt.plot(self.H["val_acc"], label = "val_acc")
        plt.title("Última precisión del entrenamiento, época: {}\n" \
            "precision val: {:.2f}%".format(self.epocas, 100*self.H["val_acc"][-1]))
        plt.xlabel("Época #")
        plt.ylabel("Precisión")
        plt.legend(loc="lower left")
        plt.savefig(ruta_imagen_precision)
        plt.figure(2) #Figura loss
        plt.plot(self.H["train_loss"], label = "train_loss")
        plt.plot(self.H["val_loss"], label = "val_loss")
        plt.title("Última función pérdida entrenamiento, época: {}\n" \
            "pérdida val: {:.2f}".format(self.epocas, self.H["val_loss"][-1]))
        plt.xlabel("Época #")
        plt.ylabel("Pérdida")
        plt.legend(loc="lower left")
        plt.savefig(ruta_imagen_perdida)

        precision = self.H["val_acc"][-1]
        perdida = self.H["val_loss"][-1]
        scheduler_state = None
        if self.scheduler:
            if getattr(self.scheduler, "state_dict", None):
                scheduler_state = self.scheduler.state_dict()
        #Guardar la ultima iteracion
        torch.save({
            'epoch': self.epocas,
            'steps': self.entrenamiento_global_step,
            'model_state_dict': self.modelo.state_dict(),
            'optimizer_state_dict': self.optimizador.state_dict(),
            'scheduler_state_dict': scheduler_state,
            "accuracy": precision,
            "loss": perdida,
            "historico": self.H
        }, ruta_guardar_ultimo)
        if self.debugger:
            self.plotter.plot(self.cockpit, show_plot = False, save_plot = True,
                save_dir = ruta_guardar_ultimo, savename = "ultimo_{}".format(self.epocas),
                plot_title = "DIALUNG - Variables último entrenamiento"
            )
        #Guardar la época con la mejor precisión y que no sea nan
        if precision > self.mejor_precision and not np.isnan(perdida):
            self.mejor_precision = precision
            if self.debugger:
                self.plotter.plot(self.cockpit, show_plot = False, save_plot = True,
                    save_dir = ruta_guardar_ultimo, savename = "mejor_{}".format(self.epocas),
                    plot_title = "DIALUNG - Variables mejor entrenamiento"
                )
            torch.save({
                'epoch': self.epocas,
                'steps': self.entrenamiento_global_step,
                'model_state_dict': self.modelo.state_dict(),
                'optimizer_state_dict': self.optimizador.state_dict(),
                'scheduler_state_dict': scheduler_state,
                "accuracy": precision,
                "loss": perdida,
                "historico": self.H
            }, ruta_guardar_mejor)
            #Guardar mejor reporte de clasificación
            with open(ruta_mejor_reporte_clasificacion, "w") as archivo_texto:
                archivo_texto.write(self.reporte_classificacion)
            plt.figure(1)
            plt.title("Mejor precisión del entrenamiento, época: {}\n" \
                "precisión val: {:.2f}%".format(self.epocas, 100*self.H["val_acc"][-1]))
            plt.savefig(ruta_imagen_mejor_precision)
            plt.figure(2)
            plt.title("Mejor función pérdida entrenamiento, época: {}\n" \
                "pérdida val: {:.4f}".format(self.epocas, self.H["val_loss"][-1]))
            plt.savefig(ruta_imagen_mejor_perdida)
        #Guardar ultimo reporte de clasificación
        with open(ruta_reporte_clasificacion, "w") as archivo_texto:
            archivo_texto.write(self.reporte_classificacion)
        plt.figure(1)
        plt.clf()
        plt.figure(2)
        plt.clf()

    def cargar_anteriores_resultados(self, historico, global_step):
        self.H = historico
        self.epocas = len(self.H)
        self.entrenamiento_global_step = global_step
        self.mejor_precision = max(historico["val_acc"]) #Cargo la mejor perdida del modelo
