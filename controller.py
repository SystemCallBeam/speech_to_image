import threading
import queue
import time
import speech_recognition as sr
from model import SpeechModel
from view import SpeechView
import requests

class SpeechController:
    def __init__(self):
        self.model = SpeechModel()
        self.view = SpeechView(self)

        self.text_queue = queue.Queue()
        self.image_queue = queue.Queue()

        self.is_recording = False

        # Start background threads
        threading.Thread(target=self.continuous_speech_recognition, daemon=True).start()
        threading.Thread(target=self.process_text_and_generate_image, daemon=True).start()
        threading.Thread(target=self.update_view_image, daemon=True).start()

    def continuous_speech_recognition(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Ready to start speaking...")
            while True:
                if self.is_recording:
                    try:
                        audio = recognizer.listen(source, phrase_time_limit=3)
                        text = recognizer.recognize_google(audio, language="en")
                        self.text_queue.put(text)
                    except sr.UnknownValueError:
                        pass  # Ignore unrecognized speech
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")

    def process_text_and_generate_image(self):
        while True:
            if not self.text_queue.empty():
                text = self.text_queue.get()
                image_url = self.generate_image(text)
                if image_url:
                    self.image_queue.put(image_url)
                    self.model.save_image_to_file(image_url, text)
                    self.model.save_text_to_file(text)

    def generate_image(self, text):
        # Use your API call here
        pass

    def update_view_image(self):
        while True:
            if not self.image_queue.empty():
                image_base64 = self.image_queue.get()
                self.view.update_image(image_base64)

    def start_recording(self):
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

    def show_past_texts(self):
        past_texts = self.model.get_past_texts()
        print("Past texts:\n", past_texts)

