
from model import recognize_speech, generate_image, save_text_and_image

def start_recording():
    # Start speech recognition
    recognize_speech.start()

def stop_recording():
    text = recognize_speech.stop()
    image_path = generate_image(text)
    save_text_and_image(text, image_path)
    return text, image_path

def get_past_entries():
    # Return a list of past text and image paths
    return recognize_speech.get_entries()
