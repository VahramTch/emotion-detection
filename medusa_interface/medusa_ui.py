import cv2
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = 0
        self.vid = None
        self.photo = None

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_open_camera = Button(window, text="Open Camera", width=25, command=self.open_camera)
        self.btn_open_camera.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_close_camera = Button(window, text="Close Camera", width=25, command=self.close_camera)
        self.btn_close_camera.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15

    def open_camera(self):
        if self.vid is None:
            self.vid = cv2.VideoCapture(self.video_source)
            self.update()
        print("Camera opened")
    
    def close_camera(self):
        if self.vid is not None:
            self.vid.release()
            self.vid = None
            self.canvas.delete("all")
        print("Camera closed")
    
    def update(self):
        if self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid is not None:
            self.vid.release()

def main():
    root = tk.Tk()
    app = CameraApp(root, "Medusa 1.0")
    root.mainloop()

if __name__ == "__main__":
    main()
