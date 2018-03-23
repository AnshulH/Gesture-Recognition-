import cv2
import numpy as np

cap = cv2.VideoCapture(0)
num = 0

while True:

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
    
    cv2.imshow('Thresholded', res)

    k = cv2.waitKey(10)
    if k == 50:
        cv2.imwrite('./train/jump/jump'+str(num)+'.jpg',+ res)
        num += 1
    if k == 49:
        cv2.imwrite('./train/nothing/nothing'+str(num)+'.jpg', res)
        num += 1
    if k == 48:
        cv2.imwrite('./train/jump/duck'+str(num)+'.jpg',+ res)
        num += 1    
    if k == 51:
        break

cv2.destroyAllWindows()
cap.release()
