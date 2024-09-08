import speech_recognition as sr
import threading
import queue
import time
import requests
import io
import base64
from dotenv import load_dotenv
import sys
from tkinter import Tk, Label
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
label = Label(root)
label.pack()

def continuous_speech_recognition():
    """Speech recognition thread function: captures audio and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Start speaking...")
        while True:
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
    prompt = f"A bright, cheerful image representing: {text}"

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
    return image_base64

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

# Start threads
threading.Thread(target=continuous_speech_recognition, daemon=True).start()
threading.Thread(target=process_text_and_generate_image, daemon=True).start()
threading.Thread(target=process_text, daemon=True).start()
threading.Thread(target=display_images, daemon=True).start()

# Start Tkinter's main loop to keep the window active
root.mainloop()
