'''Created on 10 de out de 2018 @author: Dennis '''
import cv2 ;import numpy as np ;import time ;import winsound
face_cscFull = cv2.CascadeClassifier("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\haarcascade_fullbody.xml")
face_cscLower = cv2.CascadeClassifier("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\haarcascade_lowerbody.xml")
face_cscUpper = cv2.CascadeClassifier("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\haarcascade_upperbody.xml")
vidcap = cv2.VideoCapture("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\banco_curto.mp4")
count = 0; list_first_frames = []; id_image = 0
while True:
    ret, original_frame = vidcap.read()
    if not ret: break    
    temp = original_frame
    if(len(list_first_frames) == 3): 
        ultima, penultima, antepenultima = list_first_frames[0], list_first_frames[1], list_first_frames[2]
        d1 = cv2.absdiff(antepenultima, penultima);d2 = cv2.absdiff(penultima, ultima);
        res = d1.astype(np.uint8);percentage = round((np.count_nonzero(res) / res.size) * 100, 4)             
        if ( percentage > 80): winsound.PlaySound('C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\alarme.wav', winsound.SND_FILENAME)    
        if (percentage >= 0):
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            full =   face_cscFull.detectMultiScale(image=temp, scaleFactor=1.1, minNeighbors=7)# minSize=(10,10), maxSize=(150,500))
            lower = face_cscLower.detectMultiScale(image=temp, scaleFactor=1.1, minNeighbors=4)# minSize=(10,10), maxSize=(150,500))
            upper = face_cscUpper.detectMultiScale(image=temp, scaleFactor=1.1, minNeighbors=4)#, minSize=(10,10), maxSize=(150,500))
            for (x, y, w, h) in full:
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(original_frame, 'Corpo', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0), 1)
                cv2.imwrite("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\video\\full\\full%d.jpg" % id_image, original_frame[y:y+h,x:x+w]);id_image += 1
            for (x, y, w, h) in lower:
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(original_frame, 'Inferior', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
                cv2.imwrite("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\video\\lower\\lower%d.jpg" % id_image, original_frame[y:y+h,x:x+w]);id_image += 1
            for (x, y, w, h) in upper:
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(original_frame, 'Superior', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 1)
                cv2.imwrite("C:\\Users\\Dennis\\eclipse-workspace-tcc\\Tcc\\video\\upper\\upper%d.jpg" % id_image, original_frame[y:y+h,x:x+w]);id_image += 1                
    if len(list_first_frames) <= 2:
        list_first_frames.append(temp)
    elif len(list_first_frames) == 3:
        list_first_frames = []
    count += 1;cv2.imshow('', original_frame);time.sleep(0.04);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0);cv2.destroyAllWindows()

