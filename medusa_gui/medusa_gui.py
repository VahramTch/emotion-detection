import cv2
import tkinter as tk
import os
import numpy as np
from tkinter import Button, Label, Menu
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model

# Get the directory of the current script
current_dir = os.getcwd()
# Path to the medusa_gui directory
medusa_gui_dir = os.path.join(current_dir, 'medusa_gui')
# Path to the assets_dir directory
assets_dir = os.path.join(medusa_gui_dir, 'assets')
# Path to the medusaicons_dir_gui directory
icons_dir = os.path.join(assets_dir, 'icons')
# Path to the themes_dir directory
themes_dir = os.path.join(assets_dir, 'themes')
# Path to the xml_dir directory
xml_dir = os.path.join(assets_dir, 'xml')
# Path to the h5models_dir directory
h5models_dir = os.path.join(assets_dir, 'h5models')

# Path to the green.json file inside the 'themes' folder
azure_tcl_file_path = os.path.join(themes_dir, 'azure.tcl')
dark_tcl_file_path = os.path.join(themes_dir, 'dark.tcl')
light_tcl_file_path = os.path.join(themes_dir, 'light.tcl')

# Path to the xml_file inside the 'xml' folder
xml_file_path = os.path.join(xml_dir, 'haarcascade_frontalface_default.xml')

# Path to the xml_file inside the 'icons' folder
ico_file_path = os.path.join(icons_dir, 'medusa2.ico')

# Path to the h5_file inside the 'f5models' folder
h5_model_path = os.path.join(h5models_dir, 'model_optimal.h5')

class MedusaInterface:
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

        # Create a menu bar
        self.menu_bar = Menu(window)
        window.config(menu=self.menu_bar)

        # Load the pretrained model once
        self.model = load_model(h5_model_path)

        # Create "File" menu
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Exit", command=self.window.quit)

        # Create "Help" menu
        self.help_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Settings", command=self.open_settings)
        self.help_menu.add_command(label="Help", command=self.open_help)

        # Create canvas for video feed
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Create a frame for the buttons
        self.button_frame = tk.Frame(window)
        self.button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        # Toggle button for opening and closing the camera
        self.btn_toggle_camera = Button(self.button_frame, text="Open Camera", width=20, command=self.toggle_camera)
        self.btn_toggle_camera.grid(row=0, column=0, padx=5)

        # Button to enable/disable face recognition
        self.btn_enable_face_recognition = Button(self.button_frame, text="Enable Face Recognition", width=20, command=self.toggle_face_recognition)
        self.btn_enable_face_recognition.grid(row=0, column=1, padx=5)

    def set_icon(self, icon_path):
        icon_img = Image.open(icon_path)
        icon_photo = ImageTk.PhotoImage(icon_img)
        self.window.iconphoto(False, icon_photo)

    def toggle_camera(self):
        if self.vid is None:
            self.open_camera()
        else:
            self.close_camera()

    def open_camera(self):
        if self.vid is None:
            self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            if not self.vid.isOpened():
                print("Error: Could not open camera.")
                return
            self.btn_toggle_camera.config(text="Close Camera")
            self.update()

    def close_camera(self):
        if self.vid is not None:
            self.vid.release()
            self.vid = None
            self.canvas.delete("all")
            self.btn_toggle_camera.config(text="Open Camera")

    def toggle_face_recognition(self):
        self.face_recognition_enabled = not self.face_recognition_enabled
        self.btn_enable_face_recognition.config(
            text="Disable Face Recognition" if self.face_recognition_enabled else "Enable Face Recognition"
        )
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
                if self.face_recognition_enabled:
                    for (x, y, w, h, face) in self.preprocess_frame(frame):
                        # Predict emotion
                        predictions = self.model.predict(face)
                        predicted_class = np.argmax(predictions[0])
                        emotion = self.class_labels[predicted_class]
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, emotion, (x + int(w / 10), y + int(y / 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Convert frame to RGB and display in canvas
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
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
    root.tk.call("source", azure_tcl_file_path)
    # root.tk.call("set_theme", "light")
    root.tk.call("set_theme", "dark")
    root.iconbitmap(ico_file_path)
    app = MedusaInterface(root, "Medusa", xml_file_path)

    # Set a minsize for the window, and place it in the middle
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
    y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
    root.geometry("+{}+{}".format(x_cordinate, y_cordinate - 20))

    root.mainloop()


if __name__ == "__main__":
    main()
