import cv2 as cv
from train import get_users_from_db
face_recognizer=cv.face.LBPHFaceRecognizer_create()
def authenticate_user(frame, recognizer, users):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face = cv.resize(face, (100, 100))  # Resize for consistency
        label, confidence = recognizer.predict(face)
        if confidence < 70:  # Adjust the threshold
            for user_id, name, _ in users:
                if user_id == label:
                    return name
    return None
# Main program
cap = cv.VideoCapture(0)
users = get_users_from_db()
face_recognizer.read('face_trained.yml')  # Correct usage: load the trained model into the recognizer
print("Position your face for authentication. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access the webcam.")
        break
    user = authenticate_user(frame, face_recognizer, users)
    if user:
        cv.putText(frame, f"Welcome, {user}!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv.putText(frame, "Face not recognized", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Face Authentication", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
