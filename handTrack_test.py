import cv2
import mediapipe as mp
import time
import handTrack_module as htm

prevTime = 0
currTime = 0

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    detector = htm.handDetector()
    img = detector.findHands(img)
    lmkList = detector.findPosition(img)
    
    # Calculate FPS
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    # Print fps on screen
    cv2.putText(img, 'Frames per second:'+str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2)
    
    # Displaying webcam feed
    cv2.imshow("Webcam feed", img)
    cv2.waitKey(1)