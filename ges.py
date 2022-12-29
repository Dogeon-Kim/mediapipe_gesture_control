import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def dist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2))+math.sqrt(math.pow(y1-y2,2))

compareIndex = [[18,4],[6,8],[10,12],[14,16],[18,20]]
open =[False,False,False,False,False]
gesture = [
    [True,True,True,True,True,"five"],
    [False,True,True,True,True,"four"],
    [False,True,True,True,False,"three"],
    [False,True,True,False,False,"two"],
    [False,True,False,False,False,"one"],
    [True,True,False,False,True,"SpiderMan"]
    ]

while True:
    success,img=cap.read()
    h,w,c=img.shape
    
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    '''
    image = cv2.flip(image,1)
        
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    results = hands.process(image) #RGB
        
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    '''
    results=my_hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for i in range(0,5):
                open[i]=dist(handLms.landmark[0].x,handLms.landmark[0].y,handLms.landmark[compareIndex[i][0]].x,handLms.landmark[compareIndex[i][0]].y) < dist(handLms.landmark[0].x,handLms.landmark[0].y,handLms.landmark[compareIndex[i][1]].x,handLms.landmark[compareIndex[i][1]].y)
            
            print(open)
            text_x=(handLms.landmark[0].x*w)
            text_y=(handLms.landmark[0].y*h)
            for i in range(0,len(gesture)):
                flag=True
                for j in range(0,5):
                    if(gesture[i][j]!=open[j]):
                        flag=False
                if(flag==True):
                    cv2.putText(
                        img,
                        gesture[i][5],(round(text_x)-50,round(text_y)-250), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,(51,255,51),4)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    
    cv2.imshow("HandTracking", img)
    cv2.waitKey(1)