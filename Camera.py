import cv2
import numpy as np


faceCascade = cv2.CascadeClassifier("D:\\source\\py\\Pavel\\CV_Tests\\CV_Mods\\Resouces\\haarcascade_frontalface_alt2.xml")



def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
capture.set(10, 150)



while True:
    success, img = capture.read()
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgC = cv2.Canny(img, 200, 200)

    
    """
    faces = faceCascade.detectMultiScale(imgG,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgL,(x,y),(x+w,y+h),(0,0,0),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgC,(x,y),(x+w,y+h),(0,0,0),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgG,(x,y),(x+w,y+h),(0,0,0),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
    #cv2.imshow('Video', imgC)
    #cv2.imshow('Video', img)
    """
    img1 = cv2.resize(img, (800,600))
    #cv2.imshow("Result", img)
    
    imgStack = stackImages(1,([img1,imgL],[imgC,imgG]))
    cv2.imshow('Video_Output', imgStack)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break



"""
faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/lena.png')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
faces = faceCascade.detectMultiScale(imgGray,1.1,4)
 
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 
cv2.imshow("Result", img)
cv2.waitKey(0)
"""