import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from text_to_speech import textToSpeech
import threading

mp_model = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model = load_model('model/final.h5')

lines = []
with open('model/word_spaces.txt') as f:
    lines = f.read().splitlines()

print(lines)
words = np.array(lines)

sequence = []
sentence = []

predictions = []
threshold = 0.8
                  
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    
vidCap = cv2.VideoCapture(0)
last_word = "space"
with mp_model.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
    while vidCap.isOpened():

        success, img = vidCap.read()
                        
        if not success: 
            break
                        
        # img.flags.writeable = False
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img.flags.writeable = False                  
        results = holistic.process(img)                 
        img.flags.writeable = True                  
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.face_landmarks:
            mp_drawing.draw_landmarks(img, results.face_landmarks, mp_model.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_model.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_model.HAND_CONNECTIONS)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-3:]
        
        
        if len(sequence) == 3:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                
                if res[np.argmax(res)] > threshold:
                    print(res[np.argmax(res)])
                    if len(sentence) > 0:
                        if words[np.argmax(res)] != sentence[-1]:
                            sentence.append(words[np.argmax(res)])
                            if words[np.argmax(res)] != "space":
                                # textToSpeech(words[np.argmax(res)])
                                temp = words[np.argmax(res)]
                                
                                thread = threading.Thread(target=textToSpeech, name="Downloader", args=(temp,"english"))
                                thread.start()
                    else:
                        sentence.append(words[np.argmax(res)])
                        if words[np.argmax(res)] != "space":
                                thread = threading.Thread(target=textToSpeech, name="Downloader", args=(temp,"english"))
                                thread.start()

            if len(sentence) > 5:
                sentence = []
            
            print(last_word)
            last_word = words[np.argmax(res)]


        
        cv2.imshow('OpenCV Feed', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    #stop capture
    vidCap.release()
    cv2.destroyAllWindows()
