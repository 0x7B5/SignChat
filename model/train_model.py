import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
    
def main():
    if not os.path.isdir("model_data"):
        os.mkdir("model_data")
        
    gen_path = os.path.join('model_data') 

    with open('words.txt') as f:
        lines = f.read().splitlines()

    words = np.array(lines)
    
    numSeq = 3
    frame_num = 10000
    

    for word in words:
        if not os.path.isdir("model_data/{}".format(word)):
            os.mkdir("model_data/{}".format(word))
    
        for y in range(numSeq):
            if not os.path.isdir("model_data/{}/{}".format(word,str(y))):
                os.mkdir("model_data/{}/{}".format(word,str(y)))
    
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        for word in words:
            for vid in range(numSeq):
                capture = cv2.VideoCapture("videos/{}/{}.mov".format(word, vid))
                for x in range(frame_num):
                    try:
                        success, img = capture.read()
                        
                        if not success: 
                            break
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                        img.flags.writeable = False                  
                        results = holistic.process(img)                 
                        img.flags.writeable = True                  
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        keypoints = extract_keypoints(results)
                        local_path = "model_data/{}/{}/{}.npy".format(word, str(vid), str(x))
                        print(keypoints)
            
                        if not os.path.exists(local_path):
                            with open(local_path, 'w+'): pass
            
                        npy_path = os.path.join(gen_path, word, str(vid), str(x))
                        np.save(npy_path, keypoints)
                        cv2.waitKey(1)
                    except cv2.error:
                        break
                
    capture.release()
    cv2.destroyAllWindows()
    label_map = {label:num for num, label in enumerate(words)}

    sequences, labels = [], []
    for action in words:
        for sequence in np.array(os.listdir(os.path.join(gen_path, action))).astype(int):
            window = []
            for frame_num in range(numSeq):
                res = np.load(os.path.join(gen_path, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

 
    x_train, _, y_train, _ = train_test_split(np.array(sequences), to_categorical(labels).astype(int), test_size=0.05)

    tb_callback = TensorBoard(log_dir= os.path.join('logs'))
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(3,1662)))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(words.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=4000, callbacks=[tb_callback])

    model.save('final.h5')
    model.load_weights('final.h5')

def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

if __name__ == "__main__":
    main()