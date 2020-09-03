# importing the modules
import cv2
import matplotlib.pyplot as plt
# capturing the videos
capture=cv2.VideoCapture(0)
def capturing(frame):
    cv2.imwrite("Output.jpeg",frame)
# loading the haarcascade files
face=cv2.CascadeClassifier("frontal_face.xml")
smile=cv2.CascadeClassifier("smile.xml")
# running the while loop for showing the image Frame
while True:
    # reading the captured video
    ret,frame=capture.read()
    # 
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,2),5)
        face_gray=frame[y:y+h,x:x+w]
        face_gray1=cv2.cvtColor(face_gray,cv2.COLOR_BGR2GRAY)
        smiles=smile.detectMultiScale(face_gray1,1.3,5)
        for (a,b,c,d) in smiles:
            sm_ratio = str(round(c/a, 3))
            if float(sm_ratio)>2.2:
                cv2.rectangle(frame, (a,b), (a + c, b + d), (0, 255, 2), 2)
                capturing(frame)
                quit()
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1)==13:
        break
capture.release()
cv2.destroyAllWindows()