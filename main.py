import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np
import pyautogui

cnn = load_model('my_model.h5')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        ret, img = cap.read()
        cv2.rectangle(img,(0,0),(300,300),(255,255,255),0)
        crop = img[0:300, 0:300]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
        low_range = np.array([0, 50, 80])
        upper_range = np.array([30, 200, 255])
        skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
        mask = cv2.inRange(hsv, low_range, upper_range)
        mask = cv2.erode(mask, skinkernel, iterations = 1)
        mask = cv2.dilate(mask, skinkernel, iterations = 1)
        mask = cv2.GaussianBlur(mask, (15,15), 1)
        res = cv2.bitwise_and(crop, crop, mask = mask)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(5)
        cv2.imshow('Thresholded', res)
        cv2.imwrite('res.jpg',res)

        res = cv2.imread('res.jpg')
        res = np.expand_dims(res, axis=0)
        gest = cnn.predict(res,batch_size = 1,verbose = 0)
        print(gest)
        if(gest[0][0] == 1):
            pyautogui.press('space')

cap.release()
cv2.destroyAllWindows()
