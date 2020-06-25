Step 1: Completed with Kaggle microcourse on Python. <br/>
Step 2: have done problems on HackerRank.<br>
Step 3:Completed opencv till 29 videos.<br>
Step 4: Implemented a mini project(airpin)<br>
Step 5: have implemented OCR.Additions are remaing to code<br>

### Code for mini project-airpen
```python
import cv2
import numpy as np
import keyboard

def nothing(x):
    pass


my_video=cv2.VideoCapture(0)
#cv2.namedWindow("track")
#cv2.resizeWindow("track",500,300)
#cv2.createTrackbar("u_h","track",359,359,nothing)
#cv2.createTrackbar("u_s","track",0,255,nothing)
#cv2.createTrackbar("u_v","track",0,255,nothing)

#cv2.createTrackbar("l_h","track",0,359,nothing)
#cv2.createTrackbar("l_s","track",0,255,nothing)
#cv2.createTrackbar("l_v","track",0,255,nothing)

x=[]
y=[]
#color=[[255,0,0],[0,0,255]]

while(1):
    ret, frame = my_video.read()
    frame = cv2.resize(frame, (500, 500))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #u_h = cv2.getTrackbarPos("u_h", "track")
    #u_s = cv2.getTrackbarPos("u_s", "track")
    #u_v = cv2.getTrackbarPos("u_v", "track")
    #l_h = cv2.getTrackbarPos("l_h", "track")
    #l_s = cv2.getTrackbarPos("l_s", "track")
    #l_v = cv2.getTrackbarPos("l_v", "track")

    lower=np.array([0,97,212])
    upper=np.array([29,255,255])

    mask=cv2.inRange(hsv,lower,upper)
    contours,heirarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print("contours no",len(contours))
    #print(contours)
    #print(type(contours))
    if len(contours)>0:
        a=contours[0]
        x.append(a[0][0][0])
        y.append(a[0][0][1])


    print("done....................")
    bitwise_and=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.drawContours(frame, contours, -1, (0, 0, 0), 4)




    if len(x)>=2:
        for i in range((len(x)-1)):
            frame = cv2.line(frame, (x[i], y[i]), (x[i+1], y[i+1]),(0,0,255), 3)


    cv2.imshow("my_video",frame)
    cv2.imshow("hsv",hsv)
    cv2.imshow("mask",mask)
    cv2.imshow("bitwise_and",bitwise_and)

    if cv2.waitKey(1)==27:
        break

cv2.destroyAllWindows()

```
### Code for sppech to text:simple implementation
```python
import pyaudio
import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)

r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable
text=""
with sr.Microphone() as source:

    print("Talk")
    r.adjust_for_ambient_noise(source)
    print("go")
    audio_text = r.listen(source)
    print("Time over, thanks")
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    text=text + r.recognize_google(audio_text)
    print("Text: " ,text)

if text=="read":
    print(" doing ocr/reading pdf")

if text =="right":
    print("writing in google document")
```
### Code for simple implementation using IP webcam with gtts
The input image is taken from the phone and sent to the laptop using http server.
```python

import requests
import numpy as np
import cv2
import pytesseract
import PIL

url="http://192.168.0.101:8080/shot.jpg"

while 1:
    img_resp=requests.get(url)
    img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
    img=cv2.imdecode(img_arr,-1)

    _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("img", img)
    cv2.imshow("th1", th1)
    print(pytesseract.image_to_string(th1))


    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
```
### Code for basic implementation of the final project on Windows.

```python
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS
import os
import cv2
import pytesseract
import wikipedia
import requests
import numpy as np
from skimage.filters import threshold_local
import imutils
import PyPDF2
from pyimagesearch.transform import four_point_transform


def speak(str):
    "Speaks the input string"
    tts = gTTS(str, lang='en')
    tts.save("Sound.mp3")
    playsound('Sound.mp3')


def listen():
    "Listens for speech, converts it into text and returns the text"
    r = sr.Recognizer()  # creating an instance of class Recognizer
      # Using first microphone (default) in list of mics
    with sr.Microphone() as source:
        print("talk")
        r.adjust_for_ambient_noise(source)
        print("go")
        audio = r.listen(source)
        print("done")

    input_text = r.recognize_google(audio)  # Google Web Speech API; input_text will store response of user
    return input_text


# Welcome message
speak("Hi, Do you want to read or write or search?")

# Taking input from user
input1 = listen()
print(input1)
if input1.lower() == "read" or input1.lower()== "weed":
    os.remove("Sound.mp3")
    speak("Do you want to read a PDF or read an image?")
    input2 = listen()

    if input2.lower() == "pdf":
        os.remove("Sound.mp3")
        speak("Say the name of the file you want to read?")
        name=listen()

        os.remove("Sound.mp3")
        speak("is it a pdf or a text file?")
        type_of_file=listen()

        if type_of_file == "pdf":
            pdfFileObj = open(name+".pdf", "rb")
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

            mytext = ""

            for pageNum in range(1, 3):
                pageObj = pdfReader.getPage(pageNum)
                mytext = mytext + pageObj.extractText()

            print(mytext)
            pdfFileObj.close()
            speak(mytext)

        if type_of_file == "text":
            file=open(name+".txt" ,"r")
            mytext=file.read()
            print(mytext)
            os.remove("Sound.mp3")
            speak(mytext)


    elif input2.lower() == "image":
        url = "http://192.168.0.102:8080/shot.jpg"
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        image = cv2.imdecode(img_arr, -1)

        #image = cv2.imread('//Users/avishi/Desktop/WhatsApp Image 2020-06-25 at 4.21.41 PM.jpeg')
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height=500)
        cv2.imwrite("poetry.jpeg", image)
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show the original image and the edge detected image
        print("STEP 1: Edge Detection")
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break

        # show the contour (outline) of the piece of paper
        print("STEP 2: Find contours of paper")
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 15, offset=10, method="gaussian")
        warped = (warped > T).astype("uint8") * 255

        # show the original and scanned images
        print("STEP 3: Apply perspective transform")
        cv2.imshow("Original", imutils.resize(orig, height=650))
        cv2.imshow("Scanned", imutils.resize(warped, height=650))
        cv2.waitKey(1)

        my_text = pytesseract.image_to_string(warped)

        print(my_text)
        cv2.waitKey(1)
        os.remove("Sound.mp3")
        speak(my_text)

        if cv2.waitKey(0)==27:

            cv2.destroyAllWindows()


elif input1.lower() == "write" or input1.lower() == "right":
    os.remove("Sound.mp3")
    speak("What should be the name of the file?")
    name = listen()
    name = name.lower() + ".txt"
    with open(name, 'a+') as f:
        os.remove("Sound.mp3")
        speak("Start speaking")
        f.write("\n" + listen())


elif input1.lower() == "search":
    os.remove("Sound.mp3")
    speak("What do you want to search for?")
    search=listen()
    result=wikipedia.summary(search,sentences=1)
    os.remove("Sound.mp3")
    speak("According to wikipedia,  " + result)



else:
    os.remove("Sound.mp3")
    speak("Sorry, could not understand")
    
```

The code has been tested and the result of an input image given through http server is been attached in folder as poetry.jpeg, poetrygtts.mp3 <br>

