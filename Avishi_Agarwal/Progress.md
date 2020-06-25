# Progress Report
## Week 1
Completed Kaggle Course on Python and learnt to use GitHub

## Week 2
Started solving problems on HackerRank

## Week 3
OpenCV Tutorials
## Rest of the Month
Open CV Mini Project: Invisibility Cloak
```python
import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0)
cap.set(3, 512)
cap.set(4, 512)

_, frame_ = cap.read()

background = cv2.imread('background.jpg')
background = cv2.resize(background, (frame_.shape[1], frame_.shape[0]))

cv2.namedWindow('Tracking')
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while cap.isOpened():

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    mask1 = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask1)

    res = res1 + res2

    cv2.imshow('res', res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
```


**Basic Implementation of OCR of a picture taken from phone and converting text to speech**
```python
import pytesseract
import cv2 as cv
import numpy as np
from gtts import gTTS
import os

img = cv.imread('test_image.jpeg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#_, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 20, 10)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 11)

cv.imshow("Image", img)
#cv.imshow("THRESH_BINARY", th1)
#cv.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

my_text = pytesseract.image_to_string(th3)

print(my_text)
output = gTTS(my_text)
output.save("output_maths_page1.mp3")
os.system("start output_maths_page1.mp3")

cv.waitKey(0)
```
Image used in the above code: 
![test_image](https://github.com/tejalbarnwal/ITSP_2020-Oculus/blob/master/Avishi_Agarwal/test_image.jpeg)


**Implementation of converting PDF file to text, and applying text to speech**
```python
import PyPDF2
from gtts import gTTS

pdfFileObj = open("Short-stories-from-100-Selected-Stories.pdf", "rb")
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

mytext = ""

for pageNum in range(1, 3):
    pageObj = pdfReader.getPage(pageNum)
    mytext = mytext + pageObj.extractText()

print(mytext)
pdfFileObj.close()

tts = gTTS(text=mytext, lang='en')
tts.save("story_test.mp3")
```

## BASIC IMPLEMENTION, FIRST VERSION
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

**perspective transform**
```python
# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import pytesseract

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be scanned")
# args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# image = cv2.imread(args["image"])
image = cv2.imread('//Users/avishi/Desktop/WhatsApp Image 2020-06-25 at 4.21.41 PM.jpeg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

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
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 15, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)

my_text = pytesseract.image_to_string(warped)

print(my_text)
cv2.waitKey(0)
```
