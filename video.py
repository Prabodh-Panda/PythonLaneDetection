import cv2 as cv
import numpy as np

video = cv.VideoCapture("road_car_view.mp4")

while True:
    ret,orig_frame = video.read()
    if not ret:
        video = cv.VideoCapture("road_car_view.mp4")
        continue

    frame = cv.GaussianBlur(orig_frame,(5,5),0)

    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    low_yellow = np.array([10,94,140])
    up_yellow = np.array([48,255,255])
    mask = cv.inRange(hsv,low_yellow,up_yellow)

    edges = cv.Canny(mask,75,150)
    lines = cv.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),5)

    cv.imshow("Frame",frame)
    cv.imshow("Edges",edges)

    key = cv.waitKey(25)
    if key == 27:
        #Press Escape To Exit
        break
video.release()
cv.destroyAllWindows()