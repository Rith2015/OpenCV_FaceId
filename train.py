import numpy as np
import cv2 as cv
import sqlite3
# Initialize lists to store faces and IDs
faces = []
ids = []
def get_users_from_db():
    """Fetch users with their face data from the SQLite database."""
    conn = sqlite3.connect('face_id.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, face_image FROM users")
    users = cursor.fetchall()
    conn.close()
    return users
def train_recognizer(users):
    """Train the face recognizer using the user data."""
    recognizer = cv.face.LBPHFaceRecognizer_create()

    for user_id, _, face_image_binary in users:
        # Decode face image from binary data
        face_image = cv.imdecode(np.frombuffer(face_image_binary, np.uint8), cv.IMREAD_GRAYSCALE)
        face_image = cv.resize(face_image, (100, 100))  # Resize to consistent size
        faces.append(face_image)
        ids.append(user_id)

    recognizer.train(faces, np.array(ids))
    return recognizer

if __name__ == "__main__":
    # Main program
    users = get_users_from_db()  # Fetch users from the database
    recognizer = train_recognizer(users)  # Train the recognizer
    # Save the trained model and data
    recognizer.save('face_trained.yml')
    np.save('faces.npy', np.array(faces, dtype='object'))
    np.save('ids.npy', np.array(ids))
    print("Training complete. Model and data saved.")
