import cv2
import mediapipe as mp
import numpy as np
import random

max_num_hands = 1
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('rsp\data\gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

def recognize_gesture(res):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    # Compute angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
    v2 = joint[
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
    v = v2 - v1  # [20,3]
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[
                                [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[
                                [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

    angle = np.degrees(angle)  # Convert radian to degree

    # Inference gesture
    data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(data, 3)
    idx = int(results[0][0])

    return rps_gesture.get(idx, None)

cap = cv2.VideoCapture(0)

# Computer's random gesture
computer_gesture = random.choice(list(rps_gesture.values()))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for res in result.multi_hand_landmarks:
            # Recognize gesture
            user_gesture = recognize_gesture(res)
            
            if user_gesture:
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=user_gesture.upper(), org=(org[0], org[1] + 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': user_gesture,
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)


            # Who wins?
            if rps_result:
                winner = None
                text = ''

                if user_gesture == computer_gesture:
                    text = 'Tie'
                elif (user_gesture == 'rock' and computer_gesture == 'scissors') or \
                     (user_gesture == 'paper' and computer_gesture == 'rock') or \
                     (user_gesture == 'scissors' and computer_gesture == 'paper'):
                    text = f'{user_gesture.capitalize()} wins'
                    winner = 'user'
                else:
                    text = f'{computer_gesture.capitalize()} wins'
                    winner = 'computer'

                cv2.putText(img, text='Computer: ' + computer_gesture, org=(int(img.shape[1] / 2), 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                if winner:
                    cv2.putText(img, text='Winner: ' + winner.capitalize(),
                                org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(0, 255, 0), thickness=2)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
