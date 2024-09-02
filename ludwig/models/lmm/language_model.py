from transformers import LlamaForCausalLM, AutoTokenizer

def return_llama_class_transformers():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)