# DIALUNG
## Software para detección de enfermedades pulmonares
Plataforma con una estructura para lograr identificar enfermedades pulmonares
## Lenguaje de programación y librerías utilizadas
- Python 
### Librerías Utilizadas
- Pytorch
## Fuente a utilizar de los datasets
- X-ray Pecho
- CT
# Datasets utilizados
-[CPCXR (COVID-19 Posteroanterior Chest X-Ray fused)](https://paperswithcode.com/dataset/cpcxr)
-[ChestX-ray14](https://stanfordmlgroup.github.io/projects/chexnet/)

## Objetivo de las enfermedades a detectar
- Tuberculosis
- Neumonía
- COVID-19

Label map:
```python
labels_map = {
    0: "normal",
    1: "tuberculosis",
    2: "neumonia",
    3: "covid-19"
}
```

## Requerimientos de Software
- Python3
Instalar dependencias con `pip install -r requirements.txt`

## Tareas de la Semana 1 
- [ ] Descargar el dataset asignado
- [ ] Programar en el lenguaje que más les convenga un programa para obtener los atributos del dataset, subirlo a Github si es posible
- [ ] Una vez obtenidos los atributos que contiene el dataset, comparar con nuestro dataset, haber si es posible unir los dos datasets para el uso de la clasificación
- [ ](Opcional) Tratar de validar el dataset, es decir revisar si los labels(etiquetas) coinciden con lo que se muestra la imagen o a su vez verificar que estén bien etiquetadas
- [ ] En estos días les paso que formato de datos se usa en PyTorch y ver como transformar nuestro dataset o a su vez hacer un programa para transformarlo
