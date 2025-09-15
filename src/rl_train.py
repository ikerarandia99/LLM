# src/rl_train.py
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from reward import reward_end_of_sentence
import torch
import random
import re
import os
import csv


# MODEL_NAME = 'crumb/nano-mistral'
MODEL_NAME = 'distilgpt2'

def load_prompts_from_corpus(corpus_path: str, limit: int = None):
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Separar frases por ., ! o ?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    prompts = [s.strip() for s in sentences if s.strip()]

    if limit:
        prompts = prompts[:limit]

    print(f"Loaded {len(prompts)} prompts from {corpus_path}.")
    return prompts

def main():
    # -----------------------------
    # Cargar modelo con Value Head
    # -----------------------------
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Asegurar token de padding
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.pretrained_model.resize_token_embeddings(len(tokenizer))

    # -----------------------------
    # Config PPO para CPU
    # -----------------------------
    ppo_config = PPOConfig(
        model_name=MODEL_NAME,
        learning_rate=1.41e-5,
        batch_size=8,                    # batch pequeño para CPU
        mini_batch_size=1,
        gradient_accumulation_steps=8    # simula batch grande sin usar mucha RAM
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # -----------------------------
    # Prompts iniciales
    # -----------------------------
    # prompts = [
    # "Deep learning models are powerful because",
    # "If I could travel anywhere in the world, I would go to",
    # "The biggest challenge in climate change is",
    # "The capital of Spain is"
    # ]
    prompts = [
        # Hechos / definiciones
        "The capital of Japan is",
        "Water boils at",
        "The largest planet in our solar system is",
        "The inventor of the telephone was",
        "Photosynthesis is the process by which",
        "The speed of light is approximately",
        "The currency used in Brazil is",
        "The tallest mountain in the world is",
        "The chemical symbol for gold is",
        "The human heart is responsible for",
        # Preguntas abiertas
        "Why do people celebrate birthdays",
        "What is the meaning of friendship",
        "How does the internet work",
        "Why is the sky blue",
        "What happens when you mix vinegar and baking soda",
        "Why do humans need sleep",
        "What makes music enjoyable",
        "Why do cats purr",
        "How do airplanes stay in the air",
        "Why do we dream at night",
        # Narrativos / creativos
        "Once upon a time, in a small village, there lived",
        "The dragon looked down and saw",
        "On a cold winter morning, a child found",
        "The treasure map led them to",
        "In the future, robots will be able to",
        "The hero took a deep breath and",
        "As the clock struck midnight, the castle",
        "The wizard raised his wand and",
        "When the spaceship landed, the crew discovered",
        "Long ago, in a distant kingdom, there was",
        # Personales / cotidianos
        "When I was a child, I loved to",
        "My favorite hobby is",
        "The best meal I ever had was",
        "In my free time, I usually",
        "The first time I rode a bicycle, I",
        "One of my happiest memories is",
        "When I feel tired, I like to",
        "My dream vacation would be in",
        "Every morning, I start my day by",
        "My favorite subject in school was",
        # Filosóficos / reflexivos
        "The purpose of life is often considered to be",
        "Happiness can be found in",
        "Success is usually achieved through",
        "Courage means being able to",
        "Wisdom is gained by",
        "True friendship is when",
        "Love is best described as",
        "The meaning of freedom is",
        "Kindness can change the world because",
        "Hope is important because",
        # Listas / ejemplos
        "Three colors of the rainbow are",
        "Some animals that live in the ocean are",
        "Popular programming languages include",
        "Healthy foods that I enjoy are",
        "Famous scientists in history include",
        "The main ingredients of a cake are",
        "Sports that require a ball include",
        "Countries in Europe include",
        "Famous works of art include",
        "Planets in the solar system include",
    ]

    # -----------------------------
    # Entrenamiento PPO
    # -----------------------------
    total_epochs = 200
    checkpoint_dir = "C:\Iker\LLM\src\models\gpt2-rl"
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_rewards = []
    all_losses = []

    KL_MAX = 50
    KL_MIN = -100

    for epoch in range(total_epochs):
        # Shuffle completo de prompts cada epoch
        random.shuffle(prompts)
        batch_prompts = prompts[:ppo_config.batch_size]

        # Tokenizar batch completo
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch["input_ids"]

        # -----------------------------
        # Generación vectorizada
        # -----------------------------
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=35,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=1
        )

        # -----------------------------
        # Calcular recompensas
        # -----------------------------
        rewards = []
        decoded_outputs = []
        for i, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            decoded_outputs.append(decoded)
            reward = reward_end_of_sentence(decoded,max_new_tokens=35)
            rewards.append(torch.tensor(reward, dtype=torch.float))
            print(f"[Epoch {epoch}] Prompt: {batch_prompts[i]!r}")
            print(f"Generated: {decoded!r}")
            print(f"Reward: {reward}\n")

        # -----------------------------
        # Actualizar PPO
        # -----------------------------
        queries = [q for q in input_ids]
        responses = [o for o in outputs]

        stats = ppo_trainer.step(queries, responses, rewards)

        # -----------------------------
        # Guardar métricas
        # -----------------------------
        mean_reward = float(torch.stack(rewards).mean().item())
        all_rewards.append(mean_reward)

        if "loss/policy" in stats:
            all_losses.append(stats["loss/policy"])
        else:
            all_losses.append(0.0)

        print(f"📊 Epoch {epoch+1}/{total_epochs} | reward={mean_reward:.3f} | loss={all_losses[-1]:.3f}")

        kl_value = stats.get("objective/kl", None)
        if kl_value is not None and epoch>15:
            kl_mean = kl_value.mean().item() if isinstance(kl_value, torch.Tensor) else float(kl_value)
            print(f"[Epoch {epoch}] KL divergence: {kl_mean:.3f}")
            if kl_mean > KL_MAX:
                print(f"KL divergence is too high, stopping training: {kl_mean:.3f}")
                #break
            elif kl_mean < KL_MIN:
                print(f"KL divergence is too low, stopping training: {kl_mean:.3f}")
                #break
            print(f"🔄 KL value: {kl_value:.3f} | KL_MAX: {KL_MAX:.3f} | KL_MIN: {KL_MIN:.3f}")

        # -----------------------------
        # Guardar checkpoint cada 5 epochs
        # -----------------------------
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"✅ Saved checkpoint for epoch {epoch+1}")

    # -----------------------------
    # Guardar modelo final
    # -----------------------------
    model.save_pretrained(os.path.join(checkpoint_dir, "final"))
    tokenizer.save_pretrained(os.path.join(checkpoint_dir, "final"))
    print("✅ RL model saved.")

    # -----------------------------
    # Guardar métricas en CSV
    # -----------------------------
    csv_path = os.path.join(checkpoint_dir, "training_metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_reward", "loss_policy"])
        for i, (r, l) in enumerate(zip(all_rewards, all_losses)):
            writer.writerow([i + 1, r, l])

    print(f"✅ Training metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
