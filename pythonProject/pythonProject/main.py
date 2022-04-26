# basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib -> face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils


# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)



def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB) #compute the distance between both eye corners
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e) # it will calculate the exact ratio of eye
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked then return 2 else if return 1 for drowsiness else return 0 for sleep
    if (ratio > 0.25):
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):
        return 1
    else:
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #adjust the frame

    faces = detector(gray)  #calling frontol face detector
    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()  #frame size ajustments according to Region of interest
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # initialinzing two variables for left and right eye with justified landmarks
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        #  now we will judge the eye movements and blinking and several outputs
        if (left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 4):
                status = "warning!! wake up"
                color = (255, 0, 0)

        elif (left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 4):
                status = "Alert!"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 4):
                status = "Active:)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break