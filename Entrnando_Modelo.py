#-----------------------Importar las librerias -------------
import cv2
import numpy as np
import os

#---------------------Importar las fotos tomadas anteriormente--------------
direccion = 'C:/Users/lchumi/PycharmProjects/faceDetection/Fotos'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
cont = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir # Leer las fotos de los rostros

    for fileName in os.listdir(nombre):
        etiquetas.append(cont) # Asignamos las etiquetas
        rostros.append(cv2.imread(nombre + '/' + fileName, 0))


    cont += 1

#print(etiquetas)

#----------------------Creamos el modelo ---------------
reconocimiento = cv2.face.LBPHFaceRecognizer.create()

#----------------------Entrenamos el modelo------------------
reconocimiento.train(rostros, np.array(etiquetas))

#----------------------Guardamos el modelo ------------------
reconocimiento.write("ModeloEntrenado.xml")

