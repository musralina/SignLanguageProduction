# crop the image, to reduce the dimension
import cv2
import numpy as np
import math

offset = 150
imgSize = 370


def handCrop(hand, img):
    x, y, width, height = hand['bbox']
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
 
    imgCrop = img[y - offset:y + height + offset, x - offset:x + width + offset]
    if imgCrop.size != 0:
        # cv2.imshow('imgCrop', imgCrop)
        imgCropShape = imgCrop.shape
        hwRatio = height/width
        if hwRatio > 1:
            k = imgSize/height
            widthCalc = math.ceil(k * width)
            imgResize = cv2.resize(imgCrop, (widthCalc, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - widthCalc) / 2)
            imgWhite[:, widthGap:widthCalc + widthGap] = imgResize   
        else:    
            k = imgSize / width
            heightCalc = math.ceil(k * height)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCalc))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - heightCalc) / 2)
            imgWhite[heightGap:heightCalc + heightGap, :] = imgResize           
    return imgWhite
