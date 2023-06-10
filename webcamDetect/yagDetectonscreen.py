import cv2
import numpy as np
from PIL import ImageGrab
from ultralytics import YOLO
model = YOLO("last.pt")

while True:
    # Capturar pantalla utilizando PIL
    img = ImageGrab.grab()
    # print(type(img))
    # Convertir la imagen PIL a una matriz numpy
    frame = np.array(img)

    # Convertir el espacio de color de BGR a RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = model.predict(frame)
    # model.predict(img, save=True, imgsz=320, conf=0.5)
    # Mostrar la imagen capturada
    
    # cv2.imshow('Captura de pantalla', frame)
    res = model(source=img)
    res_plotted = res[0].plot()
    cv2.imshow("result", res_plotted)
    
    

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Cerrar todas las ventanas y liberar recursos
cv2.destroyAllWindows()
