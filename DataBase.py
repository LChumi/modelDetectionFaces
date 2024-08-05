#---------------------------Importamos las librerias ---------------
import cv2
import mediapipe as mp
import os

#--------------------------Creacion de la carpeta donde alamcenaremos las fotos---------------------
nombre = 'Luis_Chumi_Mascarilla'
#nombre = 'Luis_Chumi'
direccion = 'C:/Users/lchumi/PycharmProjects/faceDetection/Fotos'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print('Carpeta creada')
    os.makedirs(carpeta)
#Iniciamos niestro contador
cont =0

#-------------------------Declaracion del detector ---------------------------------
detector = mp.solutions.face_detection #Detector
dibujo = mp.solutions.drawing_utils    #Dibujo

#-------------------------Realizar la videocamara ----------------------------------
cap = cv2.VideoCapture(0)

#-------------------------Inicializar los parametros de deteccion --------------------
with detector.FaceDetection() as rostros:

    #Iniciamos While True
    while True:
        #Lectura de la videCaptura de los fotogramas
        ret, frame = cap.read()

        #Eliminar el erro de espej
        frame = cv2.flip(frame, 1)

        #Eliminar el error de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Deteccion de rostros
        resultado = rostros.process(rgb)

        #Filtro de seguridad
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro)

                #Extraemos el ancho y el alto de nuestra ventana
                al, an, _ = frame.shape

                #Extraemos el X inicial e Y inicial
                xi = rostro.location_data.relative_bounding_box.xmin
                yi = rostro.location_data.relative_bounding_box.ymin

                #Estraemos el ancho y el alto
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                #Convertimos a pixeles
                xi = int(xi * an)
                yi = int(yi * al)
                ancho = int(ancho * an)
                alto = int(alto * al)

                #Hallamos XFinal e YFinal
                xf = xi + ancho
                yf = yi + alto

                #Extraccion de Pixeles
                cara = frame[yi:yf, xi:xf]

                #Redimencionar las fotos
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)

                #Almacenar nuestras imagenes
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(cont),cara)
                cont += 1

        #Mostramos los fotogramas
        cv2.imshow("Reconocimineto facial y de tapabocas ", frame)
        #Leyendo una tecla
        t = cv2.waitKey(1)
        if t == 27 or cont >= 300:
            break

cap.release()
cv2.destroyAllWindows()

