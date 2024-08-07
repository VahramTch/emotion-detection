import cv2
import tkinter as tk
import os
import numpy as np
from tkinter import Button, Label, Menu
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import platform

# Get the directory of the current script
current_dir = os.getcwd()
# Path to the assets directory
assets_dir = os.path.join(current_dir, 'medusa_gui', 'assets')
# Path to the icons directory
icons_dir = os.path.join(assets_dir, 'icons')
# Path to the xml directory
xml_dir = os.path.join(assets_dir, 'xml')
# Path to the h5models directory
h5models_dir = os.path.join(assets_dir, 'h5models')

# Path to the xml file inside the 'xml' folder
xml_file_path = os.path.join(xml_dir, 'haarcascade_frontalface_default.xml')

# Path to the ico file inside the 'icons' folder
ico_file_path = os.path.join(icons_dir, 'medusa2.ico')

# Path to the h5 file inside the 'h5models' folder
h5_model_path = os.path.join(h5models_dir, 'model_optimal.h5')

# Check if files exist
if not os.path.exists(xml_file_path):
    raise IOError(f"Cannot find Haar cascade XML file at {xml_file_path}")
if not os.path.exists(h5_model_path):
    raise IOError(f"Cannot find H5 model file at {h5_model_path}")
if not os.path.exists(ico_file_path):
    print(f"Icon file not found at {ico_file_path}, using default icon")


class EmotionDetectorGUI:
    def __init__(self, window, window_title, cascade_path):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = None
        self.photo = None
        self.delay = 15
        self.face_recognition_enabled = False

        # Define class labels
        self.class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Load the Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Cannot load cascade classifier from {cascade_path}")
        print(f"Loaded cascade classifier from {cascade_path}")

        # Create a menu bar
        self.menu_bar = Menu(window)
        window.config(menu=self.menu_bar)

        # Load the pretrained model once
        self.model = load_model(h5_model_path)
        print(f"Loaded model from {h5_model_path}")

        # Create "File" menu
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Exit", command=self.window.quit)

        # Create "Help" menu
        self.help_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Settings", command=self.open_settings)
        self.help_menu.add_command(label="Help", command=self.open_help)

        # Create a frame for the video feed and buttons
        self.main_frame = tk.Frame(window)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Create canvas for video feed
        self.canvas = tk.Canvas(self.main_frame, width=1920, height=1080)
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Create a frame for the buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        # Toggle button for opening and closing the camera
        self.btn_toggle_camera = Button(self.button_frame, text="Open Camera", width=20, command=self.toggle_camera)
        self.btn_toggle_camera.grid(row=0, column=0, padx=5)

        # Button to enable/disable face recognition
        self.btn_enable_face_recognition = Button(self.button_frame, text="Enable Face Recognition", width=20,
                                                  command=self.toggle_face_recognition)
        self.btn_enable_face_recognition.grid(row=0, column=1, padx=5)

        # Configure grid weights to ensure proper resizing behavior
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def toggle_camera(self):
        if self.vid is None:
            self.open_camera()
        else:
            self.close_camera()

    def open_camera(self):
        print("Attempting to open camera...")
        if self.vid is None:
            if platform.system() == 'Windows':
                self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            elif platform.system() == 'Darwin':
                self.vid = cv2.VideoCapture(self.video_source)
            else:
                'Did not find platform system'
            if not self.vid.isOpened():
                print("Error: Could not open camera.")
                self.vid = None
                return
            print("Camera opened successfully.")
            self.btn_toggle_camera.config(text="Close Camera")
            self.update()

    def close_camera(self):
        print("Closing camera...")
        if self.vid is not None:
            self.vid.release()
            self.vid = None
            self.canvas.delete("all")
            self.btn_toggle_camera.config(text="Open Camera")
            print("Camera closed.")

    def toggle_face_recognition(self):
        self.face_recognition_enabled = not self.face_recognition_enabled
        self.btn_enable_face_recognition.config(
            text="Disable Face Recognition" if self.face_recognition_enabled else "Enable Face Recognition"
        )
        print(f"Face recognition {'enabled' if self.face_recognition_enabled else 'disabled'}.")

    def preprocess_frame(self, frame, image_size=(48, 48)):
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert face to grayscale
            face = cv2.resize(face, image_size)  # Resize to 48x48
            face = face.astype('float32') / 255.0  # Normalize pixel values
            face = np.expand_dims(face, axis=-1)  # Add channel dimension
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            yield (x, y, w, h, face)

    def update(self):
        if self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                # Get the dimensions of the frame
                frame_height, frame_width = frame.shape[:2]

                # Calculate the aspect ratios
                frame_aspect = frame_width / frame_height
                canvas_aspect = 1920 / 1080

                # Determine the scaling factor and new dimensions to maintain aspect ratio
                if frame_aspect > canvas_aspect:
                    # Wider than the canvas aspect ratio
                    new_width = int(frame_height * canvas_aspect)
                    new_height = frame_height
                    start_x = (frame_width - new_width) // 2
                    start_y = 0
                else:
                    # Taller than the canvas aspect ratio
                    new_width = frame_width
                    new_height = int(frame_width / canvas_aspect)
                    start_x = 0
                    start_y = (frame_height - new_height) // 2

                # Crop the frame to maintain aspect ratio
                cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]

                # Resize the cropped frame to fit the canvas
                resized_frame = cv2.resize(cropped_frame, (1920, 1080))

                if self.face_recognition_enabled:
                    for (x, y, w, h, face) in self.preprocess_frame(resized_frame):
                        # Predict emotion
                        predictions = self.model.predict(face)
                        predicted_class = np.argmax(predictions[0])
                        emotion = self.class_labels[predicted_class]

                        # Draw rectangle and label
                        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(resized_frame, emotion, (x + int(w / 10), y + int(y / 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)

                # Convert frame to RGB and display in canvas
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.window.after(self.delay, self.update)

    def open_settings(self):
        self.show_popup("Settings")

    def open_help(self):
        self.show_popup("Help")

    def show_popup(self, title):
        popup = tk.Toplevel(self.window)
        popup.title(title)
        label = Label(popup, text="Hello World", font=("Helvetica", 16))
        label.pack(padx=20, pady=20)

    def __del__(self):
        if self.vid is not None:
            self.vid.release()


def main():
    root = tk.Tk()
    if os.path.exists(ico_file_path):
        root.iconbitmap(ico_file_path)
    else:
        print(f"Icon file not found: {ico_file_path}")
    app = EmotionDetectorGUI(root, "Emotion Detector", xml_file_path)

    # Set a minsize for the window, and place it in the middle
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
    y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
    root.geometry("+{}+{}".format(x_cordinate, y_cordinate - 20))

    root.mainloop()


if __name__ == "__main__":
    main()
