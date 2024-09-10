import speech_recognition as sr
import threading
import queue
import time
import requests
import io
import base64
from dotenv import load_dotenv
import sys
import os
from tkinter import Tk, Label, Button, Frame, filedialog
from PIL import Image, ImageTk  # Pillow for handling images

# Ensure UTF-8 output for any print statements
sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env (if needed for other parts of your project)
load_dotenv()

# Initialize queues for communication between threads
text_queue = queue.Queue()
image_queue = queue.Queue()

# Tkinter setup for displaying images
root = Tk()
root.title("Real-time Image Display")
root.geometry("800x600")

# Divide the window into two parts: one for buttons and one for images
button_frame = Frame(root)
button_frame.pack(side="top", fill="x")

image_frame = Frame(root)
image_frame.pack(side="bottom", fill="both", expand=True)

label = Label(root)
label.pack()

# Global variable to control recording
is_recording = False

# Define the text file and image directory paths
text_log_file = "recognized_texts.txt"
image_directory = "generated_images"

# Create directory if it doesn't exist
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

def continuous_speech_recognition():
    """Speech recognition thread function: captures audio and converts it to text."""
    """Capture audio only when recording is active."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ready to start speaking...")
        while True:
            if is_recording:
                try:
                    audio = recognizer.listen(source, phrase_time_limit=3)
                    text = recognizer.recognize_google(audio, language="en")
                    text_queue.put(text)
                except sr.UnknownValueError:
                    pass  # If speech was not recognized
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")

def process_text_and_generate_image():
    """Processes recognized text and generates an image based on the text."""
    while True:
        if not text_queue.empty():
            text = text_queue.get()
            
            # Generate image based on text
            image_url = generate_image(text)
            if image_url:
                image_queue.put(image_url)
                save_image_to_file(image_url, text)
                save_text_to_file(text)
                

def read_file(filepath):
    """Reads a file and returns its content."""
    with open(filepath, 'r') as file:
        return file.read().strip()

def generate_image(text):
    """Calls the Stability AI API to generate an image based on the given text."""
    api_key = read_file('stability_key.txt')  # Ensure this file exists with your API key
    api_host = "https://api.stability.ai"
    engine_id = "stable-diffusion-xl-1024-v1-0"

    # Example prompt based on recognized text
    prompt = f"Create an image with a clean white background. Use simple, thin lines to represent an abstract or conceptual idea based on the '{text}'"

    # Make the API request to generate an image
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 1024,  # Make sure dimensions are valid for the engine
            "width": 1024,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None

    # Extract image from the response
    data = response.json()
    image_base64 = data["artifacts"][0]["base64"]
    print('Return prompt:' + text)
    return image_base64

def save_text_to_file(text):
    # Save text to file
    with open(text_log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")
                    
def save_image_to_file(image_base64, text):
    """Saves the image to a file with a name based on the text."""
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    # Save image with a unique name
    image_filename = f"{text[:10]}_{int(time.time())}.png"
    image.save(os.path.join(image_directory, image_filename))
    
def display_images():
    """Thread to display images on the Tkinter screen."""
    while True:
        if not image_queue.empty():
            image_base64 = image_queue.get()
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            # Resize image to fit window
            image = image.resize((800, 600), Image.Resampling.LANCZOS)

            # Convert image to Tkinter-compatible format
            tk_image = ImageTk.PhotoImage(image)
            label.config(image=tk_image)
            label.image = tk_image  # Keep a reference to avoid garbage collection

        time.sleep(0.01)

def process_text():
    """Thread to print recognized text to the console."""
    while True:
        if not text_queue.empty():
            text = text_queue.get()
            print(f"Recognized Text: {text}")

def start_recording():
    """Starts the speech recognition process."""
    global is_recording
    is_recording = True
    print("Recording started...")

def stop_recording():
    """Stops the speech recognition process."""
    global is_recording
    is_recording = False
    print("Recording stopped.")
    
def show_past_texts():
    """Opens and displays the past recognized texts."""
    with open(text_log_file, "r", encoding="utf-8") as f:
        past_texts = f.read()
    print("Past texts:\n", past_texts)
    
def quit_app():
    root.quit()
    
# Button to quit
quit_button = Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side="right", padx=10, pady=10)

# Button to display past texts
show_texts_button = Button(button_frame, text="Show Past Texts", command=show_past_texts)
show_texts_button.pack(side="left", padx=10, pady=10)

# Buttons to start/stop recording
start_button = Button(button_frame, text="Start Recording", command=start_recording)
start_button.pack(side="left", padx=10, pady=10)

stop_button = Button(button_frame, text="Stop Recording", command=stop_recording)
stop_button.pack(side="left", padx=10, pady=10)

# Start threads
threading.Thread(target=continuous_speech_recognition, daemon=True).start()
threading.Thread(target=process_text_and_generate_image, daemon=True).start()
threading.Thread(target=process_text, daemon=True).start()
threading.Thread(target=display_images, daemon=True).start()

# Start Tkinter's main loop to keep the window active
root.mainloop()
