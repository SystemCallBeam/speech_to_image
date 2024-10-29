import gradio as gr
import speech_recognition as sr
from googletrans import Translator
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import spacy
import requests
import base64
import io

# Load models and processor
nlp = spacy.load("en_core_web_sm")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
translator = Translator()

# Function to recognize speech in Thai and translate to English
def recognize_and_translate():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            thai_text = recognizer.recognize_google(audio, language="th-TH")
            english_text = translator.translate(thai_text, src="th", dest="en").text
            return thai_text, english_text
        except Exception as e:
            return "Error: " + str(e), "Error: " + str(e)

# Generate an image based on the prompt and return the similarity score
def generate_image(english_text):
    prompt = ' '.join([token.text for token in nlp(english_text) if token.pos_ in ("NOUN", "VERB")])
    
    # Stability API request
    api_key = "YOUR_STABILITY_API_KEY"  # Replace with your key
    response = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
    )

    if response.status_code == 200:
        image_base64 = response.json()["artifacts"][0]["base64"]
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Evaluate similarity using CLIP
        inputs = clip_processor(text=[english_text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            similarity = (text_embeds @ image_embeds.T).item()

        return image, similarity
    else:
        return None, 0.0

# Gradio interface functions
def process_audio():
    thai_text, english_text = recognize_and_translate()
    image, score = generate_image(english_text)
    return thai_text, english_text, score, image

# Gradio layout with image on the right
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            thai_text = gr.Textbox(label="Thai Text")
            english_text = gr.Textbox(label="English Translation")
            score = gr.Number(label="CLIP Similarity Score")
            start_stop_button = gr.Button("Start/Stop Recording")
            start_stop_button.click(process_audio, outputs=[thai_text, english_text, score, gr.Image()])
        with gr.Column():
            image_display = gr.Image(label="Generated Image")

demo.launch()
