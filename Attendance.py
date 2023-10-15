import cv2
import face_recognition
import os
from datetime import datetime
import pyttsx3
import logging

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize a logger for error and event logging
logging.basicConfig(filename='attendance.log', level=logging.INFO)

# Function to mark attendance for known persons
def mark_attendance(name):
    try:
        if name != "Unknown":
            with open('Attendance.csv', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime('%H:%M:%S')
                f.write(f'{name},{dt_string}\n')
                log_attendance_event(name)  # Log attendance event
    except FileNotFoundError:
        log_error("File 'Attendance.csv' not found")  # Log the error
    except Exception as e:
        log_error(f'Error while marking attendance: {str(e)}')  # Log the error

# Function to retrieve records for multiple persons by names (case-insensitive)
def get_records_by_names(names):
    records = []
    try:
        with open('Attendance.csv', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2 and parts[0].lower() in [name.lower() for name in names]:
                    records.append(line.strip())
    except FileNotFoundError:
        records.append(f"File 'Attendance.csv' not found")
    except Exception as e:
        records.append(f"Error occurred: {str(e)}")

    if not records:
        for name in names:
            records.append(f"{name}, No record found")
    return records

# Function to log attendance events
def log_attendance_event(name):
    logging.info(f'Attendance marked for {name}')

# Function to log errors
def log_error(message):
    logging.error(message)

# Function to recognize faces in the current frame and play voice messages
def recognize_faces(frame, known_faces, known_names, confidence_threshold=0.6):
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if any(matches):
                confidence_values = face_recognition.face_distance(known_faces, face_encoding)
                match_index = matches.index(True)

                if confidence_values[match_index] <= confidence_threshold:
                    name = known_names[match_index]
                    engine.say("Identity confirmed, and Attendance Marked")
                else:
                    engine.say("Not Registered")
            else:
                engine.say("Not Registered")

            mark_attendance(name)

            log_attendance_event(name)  # Log attendance event
    except Exception as e:
        log_error(f'Error while recognizing faces: {str(e)}')  # Log the error

    engine.runAndWait()

def main():
    try:
        # Load known faces and other setup (not shown in this snippet)

        cap = cv2.VideoCapture(0)  # Initialize camera capture

        while True:
            ret, frame = cap.read()  # Capture a frame from the camera

            if not ret:
                break  # Exit the loop if frame capture fails

            # Call the recognize_faces function to recognize faces in the frame
            recognize_faces(frame, known_faces, known_names, confidence_threshold=0.6)

    except KeyboardInterrupt:
        # Handle a user interrupt (e.g., Ctrl+C) gracefully
        print("User interrupted the application.")
    except Exception as e:
        # Handle unexpected errors and log them
        print(f"An unexpected error occurred: {str(e)}")

    finally:
        # Clean up resources even if an error occurred
        cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close any open windows

if __name__ == "__main__":
    main()

