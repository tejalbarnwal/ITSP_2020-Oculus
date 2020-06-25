# ITSP 2020 Oculus

This project is a camera-based assistive text reader for blinds. It involves Text Extraction from the image and converting the Text to Speech using a Raspberry Pi.

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

#Project Code
```python
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS
import os
import cv2 as cv
import pytesseract
import PyPDF2


def speak(str):
    "Speaks the input string"
    tts = gTTS(str, lang='en')
    tts.save("Welcome.mp3")
    playsound('Welcome.mp3')


def listen():
    "Listens for speech, converts it into text and returns the text"
    r = sr.Recognizer()  # creating an instance of class Recognizer
    mic = sr.Microphone(0)  # Using first microphone (default) in list of mics
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    input_text = r.recognize_google(audio)  # Google Web Speech API; input_text will store response of user
    return input_text


# Welcome message
speak("Hi, Do you want to read or write?")

# Taking input from user
input1 = listen()
print(input1)
if input1.lower() == "read":
    speak("Do you want to read a PDF or read an image?")
    input2 = listen()
    
    if input2.lower() == "pdf":

        pdfFileObj = open("Short-stories-from-100-Selected-Stories.pdf", "rb")
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        mytext = ""

        for pageNum in range(1, 3):
            pageObj = pdfReader.getPage(pageNum)
            mytext = mytext + pageObj.extractText()

        print(mytext)
        pdfFileObj.close()
        speak(mytext)
        
    elif input2.lower() == "image":

        img = cv.imread('test_image.jpeg')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # _, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        # th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 20, 10)
        th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 11)

        cv.imshow("Image", img)
        # cv.imshow("THRESH_BINARY", th1)
        # cv.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
        cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

        my_text = pytesseract.image_to_string(th3)

        print(my_text)
        speak(my_text)
        cv.waitKey(0)

elif input1.lower() == "write" or input1.lower() == "right":
    speak("What should be the name of the file?")
    name = listen()
    name = name + ".txt"
    with open(name, 'a+') as f:
        speak("Start speaking")
        f.write("\n"+listen())

else:
    speak("Sorry, could not understand")
```
