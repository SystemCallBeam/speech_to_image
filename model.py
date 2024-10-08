import os
import time
import base64
from PIL import Image
import io

class SpeechModel:
    text_log_file = "recognized_texts.txt"
    image_directory = "generated_images"

    def __init__(self):
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)

    def save_text_to_file(self, text):
        with open(self.text_log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def save_image_to_file(self, image_base64, text):
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        image_filename = f"{text[:10]}_{int(time.time())}.png"
        image.save(os.path.join(self.image_directory, image_filename))
        
    def get_past_texts(self):
        with open(self.text_log_file, "r", encoding="utf-8") as f:
            return f.read()

