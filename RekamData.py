import cv2, os

cam = cv2.VideoCapture(0)
cam.set(3, 640) #ubah lebar
cam.set(4, 480) #ubah tinggi
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
wajahDir = 'datawajah'
faceID = input("Masukan NIM : ")
print ("Mohon Menunggu sampai Selesai .....")
ambilData = 1
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) #frmane, faktor skala, min
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2) #membuat kotak di wajah
        namaFile = 'wajah.' +str(faceID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+ '/' +namaFile, frame)
        ambilData += 1
        roiAbuabu = abuAbu [y:y+h, x:x+w]
        roiWarna = frame [y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuabu)
        for(xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna, (xe,ye), (xe+we, ye+he), (0,0,255), 1)

    cv2.imshow('Webcam', frame)
    #cv2.imshow('Webcam 2', abuAbu)
    k= cv2.waitKey(1) % 0xFF
    if k == 27 or k== ord('q'):
                break
    elif ambilData > 30:
        break
print("selesai mengambil data")
cam.release()
cv2.destroyAllWindows()