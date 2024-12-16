import cv2
import sqlite3
def save_face_to_db(name, face_image):
    conn = sqlite3.connect('face_id.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        face_image BLOB NOT NULL
    )
    """)
    # Convert face to binary
    _, buffer = cv2.imencode('.jpg', face_image)
    face_binary = buffer.tobytes()
    # Insert into database
    cursor.execute("INSERT INTO users (name, face_image) VALUES (?, ?)", (name, face_binary))
    conn.commit()
    conn.close()
    print(f"User '{name}' added successfully.")
# Initialize webcam for face registration
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Position your face and press 's' to save your face.")
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray_frame[y:y + h, x:x + w]
    cv2.imshow("Capture Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if len(faces) > 0:
            face = cv2.resize(face, (100, 100))  # Resize for consistency
            name = input("Enter your name: ")
            save_face_to_db(name, face)
        else:
            print("No face detected. Try again.")
        break
cap.release()
cv2.destroyAllWindows()
