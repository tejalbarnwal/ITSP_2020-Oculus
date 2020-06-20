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
