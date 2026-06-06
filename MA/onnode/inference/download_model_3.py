# download_model.py
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')

model.save_pretrained('/app/blip-model')
processor.save_pretrained('/app/blip-processor')