from PIL import Image
import gradio as gr
from transformers import AutoProcessor, BlipForConditionalGeneration

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(img):
    img_input = Image.fromarray(img)
    inputs = processor(img_input, return_tensors = "pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens = True)
    return caption

demo = gr.Interface(fn = generate_caption,
                    inputs = [gr.Image(label = "Image")],
                    outputs = [gr.Text(label = "Caption")],)

demo.launch()