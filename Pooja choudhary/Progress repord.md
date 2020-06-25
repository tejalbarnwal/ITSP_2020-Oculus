# PROGRESS.REPORT
## WEEK 1 
COMPLETED KAGGLE COURSE
## WEEK 2
SOLVED SOME PROBLEMS OF BASIC PYTHON
## WEEK 3
OPENCV TUTORIALS TILL 27
## WEEK 4
INVISIBLITY CLOAK MINI PROJECT
```python
import cv2
import numpy as np
import time
def noty(x) :
    return
cap=cv2.VideoCapture(0)
cap.set(3,512)
cap.set(4,512)
_,frame=cap.read()
cv2.imshow('background',frame)

backgnd=cv2.resize(frame, (640, 480))
cap=cv2.VideoCapture(0)
#cv2.namedWindow('tracking')
#cv2.createTrackbar('lh','tracking',0,255,noty)
#cv2.createTrackbar('ls','tracking',0,255,noty)
#cv2.createTrackbar('lv','tracking',0,255,noty)
#cv2.createTrackbar('uh','tracking',255,255,noty)
#cv2.createTrackbar('us','tracking',255,255,noty)
#cv2.createTrackbar('uv','tracking',255,255,noty)
img_array=[]
while True :
    _, img1 = cap.read()
    imghsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    imghsv=cv2.medianBlur(imghsv,35,0)
    #lh = cv2.getTrackbarPos('lh', 'tracking')
    #ls = cv2.getTrackbarPos('ls', 'tracking')
    #lv = cv2.getTrackbarPos('lv', 'tracking')
    #uh = cv2.getTrackbarPos('uh', 'tracking')
    #us = cv2.getTrackbarPos('us', 'tracking')
    #uv = cv2.getTrackbarPos('uv', 'tracking')
    ll=np.array([0,120,70])
    ul=np.array([10,255,255])
    mask1=cv2.inRange(imghsv,ll,ul)
    
    ll = np.array([170, 120, 70])
    ul = np.array([180, 255, 255])
    mask2 = cv2.inRange(imghsv, ll, ul)
    mask1=mask1+mask2
    

    mask1= cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    
    if cv2.waitKey(1)==ord('q') :
        break
    
    mask1=cv2.resize(mask1,(640,480))
    
    
    output=cv2.bitwise_and(backgnd, backgnd, mask=mask1)
    _,mask21=cv2.threshold(mask1,100,255,cv2.THRESH_BINARY_INV)
    output2= cv2.bitwise_and(img1,img1,mask=mask21)

    main=cv2.bitwise_or(output,output2)
    cv2.imshow('main',main)
    img_array.append(main)
out=cv2.VideoWriter('project_invisible_cloqak.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,(640,480))
for i in range(len(img_array)) :
    out.write(img_array[i])
out.release()
cv2.waitKey(0)


cv2.destroyAllWindows()
```
# SAMPLE CODE
```python
import speech_recognition as sr
from gtts import gTTS
import os
def output(text) :
    my_txt = text
    language = 'en'
    voice_output = gTTS(my_txt, lang=language, slow=False)
    voice_output.save('voice_output.mp3')
    os.system('start voice_output.mp3')
def listen() :
    r = sr.Recognizer()
    query = ''
    with sr.Microphone() as m:
        r.adjust_for_ambient_noise(m)
        audio = r.listen(m)
        query = query + r.recognize_google(audio, language='en-hi')
        return query

output('you want to read or write')
command_1=listen()
if command_1=='' :
    print('empty')
    output('thank you')

elif command_1=='read' :
    output('read pdf or image')
    command_2=listen()

    if command_2=='image' :
        print('read')
    elif command_2 =='file' :
        print('pdf')
elif command_1=='right' or command_1=='write' :
    output('get started')
    text_to_save=listen()
    print(text_to_save)
else :
    output("thank's for using")

```
## to connect phone camera with IP webcam
install IP web cam in phone 
```python
import requests
import cv2
import numpy as np
import ssl
import urllib
#print(dir(requests.urllib3))
#help(urllib.request.urlopen)
#ctx = ssl.create_default_context()
#ctx.check_hostname = False
#ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.43.195:8080/shot.jpg'
imgResp = urllib.request.urlopen(url)
imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
img = cv2.imdecode(imgNp, -1)
cv2.imshow('temp',cv2.resize(img,(600,400)))
q = cv2.waitKey(1)
    if q == ord("q"):
        break;
cv2.destroyAllWindows()
```



