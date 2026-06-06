from diffusers import DiffusionPipeline

DiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5').save_pretrained('/app/stable-diffusion-v1-5')
