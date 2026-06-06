# download_model.py
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")

# # Load GPT2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
# model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Save the model state
# torch.save(model.state_dict(), '/app/gpt2_large_model_weights.pth')

# # Save the model's configuration
# model.config.to_json_file('/app/gpt2_large_model_config.json')

# # Save the tokenizer
# tokenizer.save_pretrained('/app/tokenizer')

tokenizer.save_pretrained('/app/tokenizer_large')
model.save_pretrained('/app/gpt2_large_model')