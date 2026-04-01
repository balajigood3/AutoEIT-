from transformers import pipeline
import os

# Load a model specialized in paraphrasing
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Puzzler")

def generate_reference(text):
    # Prefix required by some T5 models
    input_text = f"paraphrase: {text}" 
    
    result = paraphraser(input_text, max_length=128, num_return_sequences=1)
    return result[0]['generated_text']
