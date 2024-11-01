import os
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
import numpy as np
from diffusers import StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    ).to("cuda")
pipe.tokenizer.clean_up_tokenization_spaces = False
# Load models and processor
nlp = spacy.load("en_core_web_sm")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
translator = Translator()
recognizer = sr.Recognizer()

def recognize():
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            thai_text = recognizer.recognize_google(audio, language="th-TH")
            return thai_text, audio
        except:
            return None, None

def translate(thai_text):
    try:
        english_text = translator.translate(thai_text, src="th", dest="en").text
        return english_text
    except:
        return None

def generate_prompt(text):
    if text is not None:
        return ' '.join([token.text for token in nlp(text) if token.pos_ in ("NOUN", "VERB") or token.dep_ == "dobj"])
    else:
        return None

""" def generate_image(prompt):
    # Stability API request
    with open('stability_key.txt', 'r', encoding='utf-8') as file:
        api_key = file.read().strip()
    
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
        
        return image
    else:
        return None """

def generate_image(prompt): 
    if prompt is not None:
        max_length = 77
        prompt_tokens = nlp(prompt)[:max_length]
        prompt = ' '.join([token.text for token in prompt_tokens]) + ''
        images = pipe(prompt, 
                      cfg_scale=6,
                      height=1024, # 512
                      width=1024, # 512
                      num_inference_steps=20 # 15
                      ).images[0]
        return images
    else:
        return None

def clip_score(text, image):
    if text and image is not None:
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            return (text_embeds @ image_embeds.T).item()
    else:
        return None

def save_log(thai_text, translated_text, prompt, image):
    if  thai_text and translated_text and prompt and image is not None:
        folder = "generated_images"
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_filename = f"{folder}/image_{len(os.listdir(folder)) + 1}.png"
        if image:
            image.save(image_filename)
        with open("thai_texts.txt", "a", encoding="utf-8") as f1, open("english_texts.txt", "a", encoding="utf-8") as f2, open("prompts.txt", "a", encoding="utf-8") as f3:
            f1.write(thai_text + "\n")
            f2.write(translated_text + "\n")
            f3.write(prompt + "\n")

def process_audio():
    thai_text, audio = recognize()
    english_text = translate(thai_text)
    prompt = generate_prompt(english_text)
    image = generate_image(prompt)
    score = clip_score(english_text, image)
    save_log(thai_text, english_text, prompt, image)
    return thai_text, english_text, prompt, score, image

def process_text(text):
    thai_text = text
    english_text = translate(thai_text)
    prompt = generate_prompt(english_text)
    image = generate_image(prompt)
    score = clip_score(english_text, image)
    save_log(thai_text, english_text, prompt, image)
    return '', english_text, prompt, score, image

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            recognize_button = gr.Button("Start Record")
            # audio_output = gr.Audio(label="Recorded voice")
            text_input = gr.Textbox(label="Enter Thai Text")
            text_button = gr.Button("Submit Text")

            thai_text_output = gr.Textbox(label="Recognized Thai Text")
            english_text_output = gr.Textbox(label="Translated English Text")
            prompt_output = gr.Textbox(label="Generated Prompt")

        with gr.Column():
            image_output = gr.Image(label="Generated Image")
            similarity_score_output = gr.Number(label="CLIP Similarity Score")

    recognize_button.click( #, audio_output
        process_audio,
        outputs=[thai_text_output, english_text_output, prompt_output, similarity_score_output, image_output]
    )
    text_button.click(
        process_text,
        inputs=[text_input],
        outputs=[thai_text_output, english_text_output, prompt_output, similarity_score_output, image_output]
    )

demo.launch()
