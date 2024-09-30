import base64
import io
import os
import tkinter as tk
from tkinter import Image, Label, scrolledtext
import speech_recognition as sr
import requests
import json
import threading
from googletrans import Translator
from PIL import Image, ImageTk
from transformers import T5ForConditionalGeneration, T5Tokenizer

"""
# Function to convert Thai text to English using LibreTranslate API
def translate_to_english(thai_text):
    if not thai_text.strip():  # ตรวจสอบว่าข้อความไทยไม่ว่างเปล่า
        print("No Thai text provided.")
        return ""
        
    api_url = "https://libretranslate.com/translate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "q": thai_text,
        "source": "th",
        "target": "en",
        "format": "text"
    }
    
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["translatedText"]
    else:
        print(f"Error translating text: {response.status_code}")
        print(f"Response content: {response.content}")
        return ""
"""
def translate_to_english(thai_text):
    if not thai_text.strip():
        print("No Thai text provided.")
        return ""

    translator = Translator()
    translated = translator.translate(thai_text, src='th', dest='en')
    return translated.text

# Function to recognize speech and return the recognized text in Thai
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        while recording:  # ฟังเสียงต่อเนื่องจนกว่าจะมีการหยุด
            audio = recognizer.listen(source)
            try:
                thai_text = recognizer.recognize_google(audio, language="th-TH")
                print(f"Recognized Thai: {thai_text}")
                process_recognized_text(thai_text)  # ส่งข้อความที่รู้จำไปยังฟังก์ชันประมวลผล
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

def process_recognized_text(thai_text):
    english_text = translate_to_english(thai_text)
    prompt = convert_to_prompt(english_text)
    output_box.insert(tk.END, f"Prompt: {prompt}\n")
    submit_prompt(prompt)
    save_text(thai_text, english_text)

# Function to start recording in a separate thread
def start_recording():
    global recording
    recording = True
    status_label.config(text="Recording started...")
    threading.Thread(target=recognize_speech, daemon=True).start()  # เริ่มรับเสียงใน thread ใหม่

# Function to stop recording
def stop_recording():
    global recording
    recording = False
    status_label.config(text="Recording stopped.")

# Function to simulate converting English text to an image generation prompt
def convert_to_prompt(english_text):
    return convert_to_prompt_with_t5(english_text)

def convert_to_prompt_with_t5(text):
    # โหลดโมเดล T5 และ tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # เตรียมข้อความให้พร้อมใช้กับโมเดล
    input_ids = tokenizer("Summarize the key points and Generate a presentation-style prompt for this text:" + text + ". Create the image as a simple black-and-white line drawing.", return_tensors="pt").input_ids
    
    # ใช้โมเดล T5 ในการแปลงข้อความ
    outputs = model.generate(input_ids)
    
    # แปลงผลลัพธ์กลับเป็นข้อความ
    prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return prompt

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().strip()

def generate_image(text):
    """Calls the Stability AI API to generate an image based on the given text."""
    api_key = read_file('stability_key.txt')  # Ensure this file exists with your API key
    api_host = "https://api.stability.ai"
    engine_id = "stable-diffusion-xl-1024-v1-0"

    prompt = text

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
            "height": 1024,
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
    
    # Decode the image from base64
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    
    return image

def submit_thai():
    thai_text = thai_input.get()
    if thai_text and not recording:
        english_text = translate_to_english(thai_text)
        prompt = convert_to_prompt(english_text)
        output_box.insert(tk.END, f"Prompt: {prompt}\n")
        submit_prompt(prompt)
        save_text(thai_text, english_text)
    else:
        output_box.insert(tk.END, "Thai text is empty or recording is active.\n")

# Save the Thai and English texts to separate files
def save_text(thai_text, english_text):
    with open("thai_text.txt", "a", encoding="utf-8") as thai_file:
        thai_file.write(thai_text + "\n")
    
    with open("english_text.txt", "a", encoding="utf-8") as english_file:
        english_file.write(english_text + "\n")
        
def save_image(image, folder="generated_images"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # สร้างชื่อไฟล์ที่ไม่ซ้ำกัน
    filename = os.path.join(folder, f"generated_image_{len(os.listdir(folder)) + 1}.png")
    image.save(filename)
    print(f"Saved image to {filename}")

def show_image(image):
    # Create a new window for the image
    image = image.resize((600, 600))  # ปรับขนาดภาพให้เป็น 600x600
    img = ImageTk.PhotoImage(image)

    # อัปเดตภาพใน Label ที่สร้างไว้
    image_label.config(image=img)
    image_label.image = img 

# Example usage: generate an image from a prompt and show it
def submit_prompt(prompt):
    image = generate_image(prompt)
    if image:
        show_image(image)
        save_image(image)

root = tk.Tk()
root.title("Image Generator")

# Frame สำหรับปุ่มและ input
control_frame = tk.Frame(root)
control_frame.pack(side=tk.LEFT, padx=10, pady=10)

# พื้นที่สำหรับแสดงผลรูปภาพ
image_frame = tk.Frame(root)
image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# พื้นที่แสดงผลภาพ (600x600)
image_label = Label(image_frame)
image_label.pack()

status_label = tk.Label(control_frame, text="Press 'Start Recording' to begin")
status_label.pack()

start_button = tk.Button(control_frame, text="Start Recording", command=start_recording)
start_button.pack()

stop_button = tk.Button(control_frame, text="Stop Recording", command=stop_recording)
stop_button.pack()

thai_input_label = tk.Label(control_frame, text="Manual Thai Input:")
thai_input_label.pack()

thai_input = tk.Entry(control_frame, width=50)
thai_input.pack()

submit_button = tk.Button(control_frame, text="Submit Thai", command=submit_thai)
submit_button.pack(pady=10)

output_box = scrolledtext.ScrolledText(control_frame, width=50, height=10)
output_box.pack()

exit_button = tk.Button(control_frame, text="Exit", command=root.quit)
exit_button.pack()

recording = False  # ตัวแปรสถานะการบันทึก
root.geometry("800x600")
root.mainloop()
