import os
import speech_recognition as sr
from PIL import Image
import time

# Recognize speech and store text
class RecognizeSpeech:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.text = ""
        self.entries = []

    def start(self):
        print("Recording started...")

    def stop(self):
        with self.microphone as source:
            print("Recording stopped. Processing...")
            audio = self.recognizer.listen(source)
            self.text = self.recognizer.recognize_google(audio, language="th-TH")
        return self.text

    def get_entries(self):
        return self.entries

recognize_speech = RecognizeSpeech()

# Generate image based on recognized text
def generate_image(text):
    # Dummy image generation process
    image = Image.new('RGB', (100, 100), color='blue')
    image_path = f'static/images/{int(time.time())}.png'
    image.save(image_path)
    return image_path

def save_text_and_image(text, image_path):
    entry = {"text": text, "image_path": image_path}
    recognize_speech.entries.append(entry)


