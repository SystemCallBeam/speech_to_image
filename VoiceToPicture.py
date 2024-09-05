import speech_recognition as sr
# import pyaudio
import threading
import queue
import time
import requests
import pygame
import io
import base64
from dotenv import load_dotenv
import os


import sys

sys.stdout.reconfigure(encoding="utf-8")


# Load environment variables
load_dotenv()

# Initialize Pygame for display
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Real-time Presentation")

# Initialize queues for communication between threads
text_queue = queue.Queue()
image_queue = queue.Queue()


def continuous_speech_recognition():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Start speaking...")
        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=3)
                text = recognizer.recognize_google(audio, language="th-TH")
                text_queue.put(text)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")


def process_text_and_generate_image():
    while True:
        if not text_queue.empty():
            text = text_queue.get()

            # Generate image based on text and sentiment
            image_url = generate_image(text)

            if image_url:
                image_queue.put(image_url)


def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read().strip()


def generate_image(text):
    api_key = read_file('stability_key.txt')
    api_host = "https://api.stability.ai"
    engine_id = "stable-diffusion-xl-1024-v1-0"

    # Adjust prompt based on sentiment
    prompt = f"A bright, cheerful image representing: {text}"

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
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None

    data = response.json()
    image_base64 = data["artifacts"][0]["base64"]
    return image_base64


def display_images():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if not image_queue.empty():
            image_base64 = image_queue.get()
            image_data = base64.b64decode(image_base64)
            image = pygame.image.load(io.BytesIO(image_data))
            screen.blit(pygame.transform.scale(image, (800, 600)), (0, 0))
            pygame.display.flip()

        time.sleep(0.1)


def process_text():
    while True:
        if not text_queue.empty():
            text = text_queue.get()
            print(f"Recognized Text: {text}")


# Start threads
threading.Thread(target=continuous_speech_recognition, daemon=True).start()
threading.Thread(target=process_text_and_generate_image, daemon=True).start()
threading.Thread(target=display_images, daemon=True).start()
threading.Thread(target=process_text, daemon=True).start()

# Keep the main thread alive
while True:
    time.sleep(1)
