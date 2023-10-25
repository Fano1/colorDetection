import cv2 as cv
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector    

cap = cv.VideoCapture(0)

def face_detect():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        _, frame = cap.read() 

        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.3, 5)

        for (x, y, width, height) in faces:
            cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3) #bgr and width

        cv.imshow("Camera", frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destoryAllWindows()


def faceM_detect():
    

    detector = FaceMeshDetector(maxFaces=2)
    while True:
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(faces[0])
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def hand_detect():

    detector = HandDetector(detectionCon=0.1, maxHands=2)
    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1)

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingersUp(hand2)

                # Find Distance between two Landmarks. Could be same hand or different hands
                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
        # Display
        cv.imshow("Image", img)
        cv.waitKey(1)

    cap.release()
    cv.destroyAllWindows()

def pose_detect():
    cap = cv.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
        if bboxInfo:
            center = bboxInfo["center"]
            cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)
    
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

faceM_detect()