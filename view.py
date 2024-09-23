from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk

class SpeechView:
    def __init__(self, controller):
        self.root = Tk()
        self.root.title("Real-time Image Display")
        self.root.geometry("800x600")
        
        self.controller = controller
        
        # Button frame
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side="top", fill="x")

        # Image frame
        self.label = Label(self.root)
        self.label.pack()

        # Start/Stop buttons
        self.start_button = Button(self.button_frame, text="Start Recording", command=self.controller.start_recording)
        self.start_button.pack(side="left", padx=10, pady=10)

        self.stop_button = Button(self.button_frame, text="Stop Recording", command=self.controller.stop_recording)
        self.stop_button.pack(side="left", padx=10, pady=10)

        # Button to show past texts
        self.show_texts_button = Button(self.button_frame, text="Show Past Texts", command=self.controller.show_past_texts)
        self.show_texts_button.pack(side="left", padx=10, pady=10)

        # Quit button
        self.quit_button = Button(self.button_frame, text="Quit", command=self.root.quit)
        self.quit_button.pack(side="right", padx=10, pady=10)

    def update_image(self, image_base64):
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(image)
        self.label.config(image=tk_image)
        self.label.image = tk_image  # Keep reference to avoid garbage collection

    def run(self):
        self.root.mainloop()
