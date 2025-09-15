from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt: str, model_name_or_path: str = "distilgpt2", max_new_tokens: int = 50, device: int = -1):
    """
    Generates text given a prompt using any HuggingFace causal LM.

    Args:
        prompt: The text prompt to generate from.
        model_name_or_path: Model name or path (e.g., 'distilgpt2', 'gpt2', or path to RL-trained model)
        max_new_tokens: Maximum number of tokens to generate.
        device: -1 for CPU, 0 for GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        device=device
    )

    result = generator(prompt)[0]["generated_text"]
    return result


if __name__ == "__main__":
    # Example usage
    prompt = "In which country is pelota played mostly?"
    output = generate_text(prompt, model_name_or_path="distilgpt2")
    print(output)
