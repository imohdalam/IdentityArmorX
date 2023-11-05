import logging
import os
import tkinter as tk
from datetime import datetime

import cv2
import face_recognition
import pyttsx3
import pytz
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize a logger for error and event logging
logging.basicConfig(filename='attendance.log', level=logging.INFO)


# Function to mark attendance for known persons with date, day, and IST
def mark_attendance(name):
    try:
        if name != "Unknown":
            now = datetime.now(pytz.timezone('Asia/Kolkata'))  # Get current time in IST
            date = now.strftime('%Y-%m-%d')
            day = now.strftime('%A')
            time = now.strftime('%H:%M:%S')
            with open('Attendance.csv', 'a') as f:
                f.write(f'{name},{date},{day},{time}\n')
                log_attendance_event(name)  # Log attendance event
    except (FileNotFoundError, PermissionError) as e:
        log_error(f"Error while marking attendance: {str(e)}")  # Log the error
    except Exception as e:
        log_error(f'Unexpected error while marking attendance: {str(e)}')  # Log the error


# Function to log attendance events
def log_attendance_event(name):
    logging.info(f'Attendance marked for {name}')


# Function to log errors
def log_error(message):
    logging.error(message)


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
                    message = f"Attendance marked for {name}"
                    voice_thread = threading.Thread(target=play_voice_message, args=(message,))
                    voice_thread.start()
                    show_popup_message(message)
                else:
                    message = f"{name} is not registered"
                    voice_thread = threading.Thread(target=play_voice_message, args=(message,))
                    voice_thread.start()
                    show_popup_message(message)
            else:
                message = "Not Registered"
                voice_thread = threading.Thread(target=play_voice_message, args=(message,))
                voice_thread.start()
                show_popup_message(message)

            mark_attendance(name)
            log_attendance_event(name)  # Log attendance event
    except Exception as e:
        log_error(f'Error while recognizing faces: {str(e)}')  # Log the error


# Function to display a popup message using Tkinter
def show_popup_message(message):
    popup = tk.Tk()
    popup.title("Face Recognition")

    # Get the screen width and height
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    # Calculate the position to center the window
    x = (screen_width - 300) // 2  # Adjust 300 to the desired width
    y = (screen_height - 600) // 2  # Adjust 600 to the desired height

    # Set the window size and position
    popup.geometry(f"300x600+{x}+{y}")

    label = tk.Label(popup, text=message, font=("Arial", 24))  # Adjust the font and size as needed
    label.pack(fill="both", expand=True, padx=20, pady=20)  # Center the label

    popup.after(4000, popup.destroy)  # Close the popup after 4 seconds

    popup.mainloop()



# Function to play a voice message
def play_voice_message(message):
    engine.say(message)
    engine.runAndWait()


def main():
    cap = None
    try:
        # Load known faces and names from the "known_faces" directory
        known_faces = []
        known_names = []
        known_faces_dir = 'known_faces'

        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Load the image using face_recognition
                image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_faces.append(encoding[0])  # Assuming there's only one face per image
                    known_names.append(os.path.splitext(filename)[0])  # Use the filename as the name

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
    except (FileNotFoundError, PermissionError) as e:
        log_error(f"Error: {str(e)}")
    except Exception as e:
        # Handle unexpected errors and log them
        log_error(f"An unexpected error occurred: {str(e)}")

    finally:
        if cap is not None:
            cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close any open windows


if __name__ == "__main__":
    main()
