import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_conf=0.5,min_track_conf=0.5):
        self.static_image_mode = static_image_mode # self -> Create an obj. It'll have it's own variable. This is that variable
        self.max_num_hands=max_num_hands
        self.min_detection_conf=min_detection_conf
        self.model_complexity = model_complexity
        self.min_track_conf=min_track_conf
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,self.model_complexity,self.min_detection_conf,self.min_track_conf) # Using default parameters here
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        #print(results.multi_hand_landmarks) # Checking values of results in real-time (in the terminal)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img 
                
    def findPosition(self, img, handNo=0, draw= True):
        lmkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lmk in enumerate(myHand.landmark):
                #print(id, lmk) # Print landmark (x,y,z) in img & id of the pt. -- Here, lmk are shown in decimals (normalized)

                # To get pxl values: 
                h, w, c = img.shape # height, width & channels
                cx, cy = int(lmk.x*w), int(lmk.y*h)
                #print(id, cx, cy) # prints lmks in pxl a/w corrp. id's
                lmkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
            
        return lmkList

def main():
    # Calculating FPS
    prevTime = 0
    currTime = 0
    
    cap = cv2.VideoCapture(0)
    detector = handDetector() # Taking default parameters here
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        
        lmkList = detector.findPosition(img)
        if len(lmkList) !=0:
            print(lmkList[0])
        
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
    
        # Displaying fps on screen
        cv2.putText(img,"FPS:"+str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        cv2.imshow("Webcam feed", img)
        cv2.waitKey(1)
        
if __name__ == '__main__': # Basically checking if we're running this particular script
    main()