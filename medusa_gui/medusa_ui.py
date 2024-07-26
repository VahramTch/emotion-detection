import cv2
import tkinter as tk
import os
from tkinter import Button, Label, Menu, ttk
from PIL import Image, ImageTk


# Get the directory of the current script
current_dir = os.getcwd()
# Path to the medusa_gui direcotry
medusa_gui_dir = os.path.join(current_dir,'medusa_gui')
# Path to the assets_dir direcotry
assets_dir = os.path.join(medusa_gui_dir,'assets')
# Path to the medusaicons_dir_gui direcotry
icons_dir = os.path.join(assets_dir,'icons')
# Path to the themes_dir direcotry
themes_dir = os.path.join(assets_dir,'themes')
# Path to the xml_dir direcotry
xml_dir = os.path.join(assets_dir,'xml')

# Path to the green.json file inside the 'themes' folder
azure_tcl_file_path = os.path.join(themes_dir,'azure.tcl')
dark_tcl_file_path = os.path.join(themes_dir,'dark.tcl')
light_tcl_file_path = os.path.join(themes_dir,'light.tcl')

# Path to the xml_file inside the 'xml' folder
xml_file_path = os.path.join(xml_dir,'haarcascade_frontalface_default.xml')

# Path to the xml_file inside the 'icons' folder
ico_file_path = os.path.join(icons_dir,'medusa2.ico')



class CameraApp:
    def __init__(self, window, window_title, cascade_path):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = None
        self.photo = None
        

        # Load the Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Create a menu bar
        self.menu_bar = Menu(window)
        window.config(menu=self.menu_bar)

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
        self.canvas.grid(row=1, column=0, columnspan=2)

        # Buttons to open and close camera
        self.btn_open_camera = Button(window, text="Open Camera", width=25, command=self.open_camera)
        self.btn_open_camera.grid(row=2, column=0, pady=10)
        
        self.btn_close_camera = Button(window, text="Close Camera", width=25, command=self.close_camera)
        self.btn_close_camera.grid(row=2, column=1, pady=10)


    def set_icon(self, icon_path):
        icon_img = Image.open(icon_path)
        icon_photo = ImageTk.PhotoImage(icon_img)
        self.window.iconphoto(False, icon_photo)

    def open_camera(self):
        if self.vid is None:
            self.vid = cv2.VideoCapture(self.video_source)
            self.update()
    
    def close_camera(self):
        if self.vid is not None:
            self.vid.release()
            self.vid = None
            self.canvas.delete("all")
    
    def update(self):
        if self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

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
    root.tk.call("source",azure_tcl_file_path)
    #root.tk.call("set_theme", "light")
    root.tk.call("set_theme", "dark")
    root.iconbitmap(ico_file_path)
    app = CameraApp(root, "Medusa", xml_file_path)

    # Set a minsize for the window, and place it in the middle
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
    y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
    root.geometry("+{}+{}".format(x_cordinate, y_cordinate-20))

    root.mainloop()

if __name__ == "__main__":
    main()

