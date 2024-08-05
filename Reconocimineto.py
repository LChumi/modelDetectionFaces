#------------------------   Importar las librerias ----------------
import cv2 
import os 
import mediapipe as mp

#--------------------Importar los nombres de las carpetas -----------------------
direccion ='C:/Users/lchumi/PycharmProjects/faceDetection/Fotos'
etiquetas = os.listdir(direccion)
print("Nombres: ", etiquetas)

#-------------------llamar al modelo entrenado-------------------
modelo = cv2.face.LBPHFaceRecognizer.create()

#---------------------Leer el modelo entrenado-----------------
modelo.read('ModeloEntrenado.xml')

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
        copia = frame.copy()

        #Eliminar el erro de espej
        frame = cv2.flip(copia, 1)

        #Eliminar el error de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia2 = rgb.copy()

        #Deteccion de rostros
        resultado = rostros.process(copia2)

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
                cara = copia2[yi:yf, xi:xf]

                #Redimencionar las fotos
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)
                cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)

                #Realizar la prediccion
                prediccion = modelo.predict(cara)

                #Vamos a mostrar los resultado en pantalla
                if prediccion[0] == 0:
                    cv2.putText(frame, '{}'.format(etiquetas[0]), (xi, yi -5), 1,1.3, (0,0,255) , 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf, yf), (0,0,255), 2)
                elif prediccion[0] == 1:
                    cv2.putText(frame, '{}'.format(etiquetas[1]), (xi, yi -5), 1,1.3, (0,0,255) , 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf, yf), (255,0,0), 2)



        #Mostramos los fotogramas
        cv2.imshow("Reconocimineto facial y de tapabocas ", frame)
        #Leyendo una tecla
        t = cv2.waitKey(1)
        if t == 27:
            break

cap.release()
cv2.destroyAllWindows()

