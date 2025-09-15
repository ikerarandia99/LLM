import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *

# Cargar modelos
original_model_id = "distilgpt2"
rl_model_path = os.path.join(MODEL_DIR, "gpt2-rl/final/")  # ruta local donde guardaste tu modelo entrenado

tokenizer = AutoTokenizer.from_pretrained(original_model_id)
original_model = AutoModelForCausalLM.from_pretrained(original_model_id)
rl_model = AutoModelForCausalLM.from_pretrained(rl_model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
original_model.to(device)
rl_model.to(device)

# Prompts de prueba
prompts = [
    "The first law of thermodynamics states that",
    "Deep learning models are powerful because",
    "If I could travel anywhere in the world, I would go to",
    "The biggest challenge in climate change is",
    "My favorite memory from school is",
    "Spain is a very"
]


# Función de inferencia
def generate_text(model, prompt, max_new_tokens=35):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Comparar outputs
for prompt in prompts:
    print("="*80)
    print(f"Prompt: {prompt}\n")
    
    original_output = generate_text(original_model, prompt)
    rl_output = generate_text(rl_model, prompt)
    
    print("🔹 GPT-2 original:")
    print(original_output, "\n")
    
    print("🔹 GPT-2 RL (entrenado):")
    print(rl_output, "\n")
