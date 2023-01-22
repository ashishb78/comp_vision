import cv2
import mediapipe as mp
import time # To count fps

cap = cv2.VideoCapture(0)

# Using a pre-defined module to detect and work on hands. Later, we'd create and use our own
mpHands = mp.solutions.hands
hands = mpHands.Hands() # Using default parameters here
mpDraw = mp.solutions.drawing_utils # To draw and map points on hand

# Calculating FPS
prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    # We need to send 'RGB' img to hands obj. (cv2 uses BGR format)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) # sending RGB img to hands obj, thus creating new object
    
    #print(results.multi_hand_landmarks) # Checking values of results in real-time (in the terminal)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lmk in enumerate(handLms.landmark):
                #print(id, lmk) # Print landmark (x,y,z) in img & id of the pt. -- Here, lmk are shown in decimals (normalized)
                
                # To get pxl values: 
                h, w, c = img.shape # height, width & channels
                cx, cy = int(lmk.x*w), int(lmk.y*h)
                print(id, cx, cy) # prints lmks in pxl a/w corrp. id's
                
                # Now, to get pxl lmk of any pt:
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
            
            mpDraw.draw_landmarks(
                img, # We pass BGR img here 'cuz we're displaying BGR afterall
                handLms, # For each hand
                mpHands.HAND_CONNECTIONS # Draw connections b/w pt.s
            )
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    
    # Displaying fps on screen
    cv2.putText(
        img, # src
        "FPS:"+str(int(fps)), # Text to be displayed
        (10,70), # Position (exp a little)
        cv2.FONT_HERSHEY_COMPLEX, # Font
        1, # Font Scale (by what no font be multiplied)
        (0,0,255), # colour (B,G,R)
        3 # Thickness
    )
    
    cv2.imshow("Webcam feed", img)
    cv2.waitKey(1)