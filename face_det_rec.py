import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize GUI window
window = tk.Tk()
window.title("Face Detection GUI")
window.geometry("800x600")

# Create label to show video
label = Label(window)
label.pack()

# Open webcam
cap = cv2.VideoCapture(0)

def detect_faces():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert to ImageTk for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Repeat
    label.after(10, detect_faces)

# Start the detection loop
detect_faces()

# Start GUI event loop
window.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
