# download_model.py
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Load GPT2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# Save the model state
# torch.save(model.state_dict(), '/app/gpt2_model_weights.pth')

# # Save the model's configuration
# model.config.to_json_file('/app/gpt2_model_config.json')

# Save the tokenizer
tokenizer.save_pretrained('/app/tokenizer')
model.save_pretrained('/app/gpt2_model')