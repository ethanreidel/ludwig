import os

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    if model_name_or_path is not None and 'lora' not in model_name_or_path:
        #lmmforconditionalgeneration here loaded
        model = 

