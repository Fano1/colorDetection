import cv2 as cv 

cap = cv.VideoCapture(0)

def face_detect():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        _, frame = cap.read() 

        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.5, 3)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        for (x, y, width, height) in faces:
            cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3) #bgr and width
            centreX = int(x + (width/2))
            centreY = int(y + (height/2))
            cv.circle(frame, (centreX, centreY), 5 ,(255, 0, 0), 2) 

            try:
                h, s, v = hsv[centreX, centreY]
                print([h , s, v]) 
            except Exception as e:
                print("Error")

            if h<20 :
                cv.putText(frame, "white", (50, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (0 ,0 ,0), 3)
            else:
                cv.putText(frame, "black", (50, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (0 ,0 ,0), 3)

        cv.imshow("Camera", frame)
        

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destoryAllWindows()

face_detect()