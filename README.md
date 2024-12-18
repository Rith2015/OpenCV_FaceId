# Face Recognition System with SQLite Integration
This project is a face recognition system that allows users to save their face data in an SQLite database, train a model to recognize faces, and authenticate users in real-time.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [File Descriptions](#file-descriptions)
6. [Usage](#usage)

## Overview
### This project demonstrates how to:
1. Capture and store face data with a webcam.
2. Train a facial recognition model using the Local Binary Pattern Histogram (LBPH) algorithm.
3. Perform face authentication based on the trained model and the database.

## Features
- Register user faces and names.
- Store user data securely in an SQLite database.
- Train an LBPH face recognizer with saved data.
- Real-time face authentication with a webcam.
 
## Requirements
    - Python 3.7+
    - OpenCV
    - NumPy
    - SQLite3
## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
2. Install the required Python packages using one of the following methods:

    ```bash
    pip install -r requirements.txt
- Or, by manually installing the necessary packages:

    ```bash
    pip install opencv-python opencv-contrib-python numpy

## File Descriptions
- **`save_face`**.py:
Captures a user's face using a webcam and stores it in an SQLite database along with the user's name.

- **`train`**.py:
Fetches data from the SQLite database, trains the LBPH face recognizer, and saves the model.

- **`login`**.py:
Authenticates users by recognizing their face with the trained LBPH model.

- **`face_id`**.db:
SQLite database containing user information (automatically created).

- **`face_trained`**.yml:
Saved LBPH facial recognition model (generated by train.py).

- **`faces.npy, ids`**.npy:
Arrays of face images and IDs used for training (saved by train.py).
## Usage
### 1. Saving Faces

#### Run save_face.py to register a user's face and name:

    python save_face.py

#### Position your face in front of the camera, and press s to save the face and input your name. 

The face data and name will be stored in face_id.db.

### 2. Training the Recognizer
#### Run train.py to train the facial recognition model:

    python train.py

This will fetch face data from the SQLite database and train the model.

The trained model will be saved as face_trained.yml.

### 3. Authenticating Users
#### Run login.py to authenticate users:

    python Login.py

Position your face in front of the camera for recognition.

The system will match the detected face with the trained model and display a greeting if recognized.

---
