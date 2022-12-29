import cv2
import mediapipe as mp 
import math

FRAME_DELAY = 100

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands

mp_fingers = mp_hands.HandLandmark


def run():
    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        max_num_hands =5,
        model_complexity = 0,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    
    while cap.isOpened():
        #모든 비디오 장치의 목록을 받아보고 
        success, image = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue
        
        image = cv2.flip(image,1)
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image) #RGB
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        width, height, _ = image.shape
        #print(width, height)
        
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                c=get_angle(
                    hand_landmark.landmark[mp_fingers.INDEX_FINGER_MCP],
                    hand_landmark.landmark[mp_fingers.INDEX_FINGER_TIP],
                    hand_landmark.landmark[mp_fingers.MIDDLE_FINGER_TIP])
                
                index_finger_tip = hand_landmark.landmark[mp_fingers.INDEX_FINGER_TIP]
                
                cv2.putText(
                    image,
                    text=f'{str(int(index_finger_tip.x*width))}, {str(int(index_finger_tip.y*height))}',
                    org=(100,100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=2
                )
                
                #result 과정처리 
                
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(FRAME_DELAY)
    cap.release()

#new*
def get_angle(ps,p1,p2):
    
    angle1 = math.atan((p1.y -ps.y) / (p1.x -ps.x))
    angle2 = math.atan((p2.y -ps.y) / (p2.x -ps.x))
    
    angle = abs(angle1 - angle2) * 180 / math.pi
    print(f'angle: {angle}')
    return angle
    #exit()
    
run()