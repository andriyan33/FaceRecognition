import cv2, os, numpy as np

wajahDir = 'datawajah'
latihDir = 'latihwajah'
cam = cv2.VideoCapture(0)
cam.set(3, 640) #ubah lebar
cam.set(4, 480) #ubah tinggi
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_DUPLEX

id = 0
names = ['Tidak Diketahui', 'acep', 'agra']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5, minSize=(round(minWidth), round(minHeight)), ) #frmane, faktor skala, min
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2) #membuat kotak di wajah
        id, confidance = faceRecognizer.predict(abuAbu[y:y+h, x:x+w]) #ketepatan
        if confidance <= 50 :
            nameID = names[id]
            confidanceTxt = " {0}%".format(round(100-confidance))
        else:
            nameID = names[0]
            confidanceTxt = " {0}%".format(round(100 - confidance))
        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(confidanceTxt), (x + 5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('FaceRecognition', frame)
    #cv2.imshow('Webcam 2', abuAbu)
    k= cv2.waitKey(1) % 0xFF
    if k == 27 or k== ord('q'):
                break
print("Selesai")
cam.release()
cv2.destroyAllWindows()