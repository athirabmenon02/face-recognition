
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input(' Enter user_id')

print("\n  please wait initializing on progress ")

count = 0

while(True):

    ret, imag = cam.read()
    
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(imag, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1


        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', imag)

    k = cv2.waitKey(100) & 0xff 
    if k == 30:
        break
    elif count >= 100: 
         break


print("\n Exiting Program......")
cam.release()
cv2.destroyAllWindows()


