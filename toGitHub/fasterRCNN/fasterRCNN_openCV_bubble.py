import cv2 as cv
import sys
import os.path
import glob
import numpy as np

# Root directory of the project
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library

confThreshold = 0.3
modelWeights = sys.argv[1]
textGraph = sys.argv[2]
imageFileDir = sys.argv[3]

cvNet = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)

images = [cv.imread(file) for file in glob.glob(os.path.join(imageFileDir, "*jpg"))]
print(len(images))

if not os.path.isdir(os.path.join(ROOT_DIR, 'processedImage')):
    os.makedirs('processedImage')
processedImagePath = os.path.join(ROOT_DIR, 'processedImage')


def drawBox(frame, info):
    rows = frame.shape[0]
    cols = frame.shape[1]
    for detection in info[0, 0, :, :]:
        score = float(detection[2])
        if score > confThreshold:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            if left <= 1 or top <= 1 or right >= (cols-1) or bottom >= (rows-1):
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (230, 0, 0), thickness=2)
            else:
                cv.circle(frame, (int((right+left)/2), int((top+bottom)/2)), int((bottom-top)/2), (0, 230, 0),
                          thickness=2)
    return frame


i = 0
for img in images:
    cvNet.setInput(cv.dnn.blobFromImage(img, swapRB=True, crop=False))
    cvOut = cvNet.forward()
    processedImage = drawBox(img, cvOut)
    cv.imshow('processedImage', processedImage)
    cv.imwrite(os.path.join(processedImagePath, 'image{}.jpg'.format(i)), processedImage)
    i += 1
    k = cv.waitKey(20000)
    if k == ord('q'):
        cv.destroyAllWindows()
        break
    elif k == ord('d'):
        k = cv.waitKey(10)


