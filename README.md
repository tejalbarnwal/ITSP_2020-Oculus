# ITSP 2020 Oculus

This project is a camera-based assistive text reader for blinds. It involves Text Extraction from the image and converting the Text to Speech, with other functionalities like converting text in a pdf of text file into audio, converting speech into text and storing it in a text file, and searching wikipedia.

The link to the proposal: https://docs.google.com/document/d/12q8lEGWqXQ-AYYyNfnh2pguzVVLyLeU-5o1twalN61E/edit?usp=sharing
## Weekly Targets

### Step 1 (7 May - 14 May)

* Complete Python microcourse on Kaggle

### Step 2 (14 May - 29 May)

* Complete problems on Hacerrank

### Step 3 (30 May - 8 June)

* Complete Opencv

### Step 4 (9 June - 11 June)

* Implement a mini project using OpenCV

### Step 5 (12 June - 25 June)

* Complete the final project

# Pseudo Code for the Project
# Project Code
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
    tts.save("Welcome.mp3")
    playsound('Welcome.mp3')


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
speak("Hi, Do you want to read, write or search?")

# Taking input from user
input1 = listen()
print(input1)
if input1.lower() == "read" :
   
    speak("Do you want to read an existing file or read an image?")
    input2 = listen()

    if input2.lower() == "existing file":
        
        speak("Say the name of the file you want to read?")
        name=listen()
        
        speak("Is it a PDF or a text file?")
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
        speak(my_text)

        if cv2.waitKey(0)==27:

            cv2.destroyAllWindows()


elif input1.lower() == "write" or input1.lower() == "right":
    
    speak("What should be the name of the file?")
    name = listen()
    name = name.lower() + ".txt"
    with open(name, 'a+') as f:
        speak("Start speaking")
        f.write("\n" + listen())


elif input1.lower() == "search":
    speak("What do you want to search?")
    search=listen()
    result=wikipedia.summary(search,sentences=1)
    speak("According to wikipedia,  " + result)



else:
    speak("Sorry, could not understand")
```
