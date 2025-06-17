import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load training images
path = 'Training_images'
images = []
classNames = []

myList = os.listdir(path)
print("Training images found:", myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Warning: Unable to read image {cl}")

print("Class names:", classNames)

# Encode faces
def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
            print(f"[{idx}] Face encoded successfully.")
        else:
            print(f"[{idx}] No face found in image!")
    return encodeList

# Mark attendance
def markAttendance(name):
    filename = 'Attendence.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Time\n")

    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'{name},{dtString}\n')
            print(f" Attendance marked for {name} at {dtString}")
        else:
            print(f" {name} already marked.")

# Main code
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    print(f"🧠 Faces detected: {len(facesCurFrame)}")

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if faceDis.size == 0:
            continue

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"Match found: {name}")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" Webcam closed by user.")
        break

cap.release()
cv2.destroyAllWindows()
