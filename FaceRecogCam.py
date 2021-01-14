import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) #ubah lebar
cam.set(4, 480) #ubah tinggi
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) #frmane, faktor skala, min
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2) #membuat kotak di wajah
    cv2.imshow('Webcam', frame)
    #cv2.imshow('Webcam 2', abuAbu)
    k= cv2.waitKey(1) % 0xFF
    if k == 27 or k== ord('q'):
                break
cam.release()
cv2.destroyAllWindows()