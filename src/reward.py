# reward.py
import re
from collections import Counter

def reward_end_of_sentence(text: str, max_new_tokens: int = 25) -> float:
    """
    Reward estable y robusto:
      - penaliza saltos de línea finales (por número)
      - comprueba el carácter de cierre real (ignorando espacios/newlines)
      - recompensa longitud suave hacia max_new_tokens
      - penaliza repeticiones consecutivas y n-grams (2-3)
      - penaliza finales con comillas o caracteres raros y 'spam' típico
      - normaliza el reward dentro de [-20, 20]
    """
    # entrada vacía o solo espacios/newlines -> castigo claro
    if text is None or text.strip() == "":
        return -10.0

    reward = 0.0

    # ------ contar saltos de línea consecutivos al final ------
    m = re.search(r'(\n+)$', text)
    num_newlines = len(m.group(1)) if m else 0

    # ------ carácter efectivo al final (ignorando whitespace/newlines) ------
    trimmed = re.sub(r'\s+$', '', text)  # solo para inspección del carácter final
    if trimmed == "":
        return -10.0
    effective_end = trimmed[-1]

    # ------ tokens / palabras (no elimina \n del texto original, solo cuenta tokens) ------
    words = re.findall(r'\S+', text)  # tokens separados por whitespace
    n_words = len(words)

    # ------ cierre correcto (basado en effective_end) ------
    if effective_end in ".!?":
        reward += 10.0
    else:
        reward -= 5.0

    # ------ penalización por saltos de línea finales (proporcional, con cap) ------
    if num_newlines > 0:
        newline_penalty = min(2.0 * num_newlines, 20.0)  # 2 puntos por newline, cap 20
        reward -= newline_penalty

    # ------ caracteres finales no deseados (comillas, etc.) ------
    bad_final_chars = {"'", "’", "”", '"', "›", ";", ":", "-", "_", "�", ")", ","}
    if effective_end in bad_final_chars:
        reward -= 5.0

    # ------ bonus por longitud (suave, lineal hasta max_new_tokens) ------
    length_ratio = min(n_words / float(max_new_tokens), 1.0)
    reward += 5.0 * length_ratio  # hasta +5 por acercarse al objetivo

    # pequeños ajustes por longitud extrema
    if n_words < 3:
        reward -= 3.0
    if n_words > max_new_tokens * 1.5:
        reward -= 3.0

    # ------ repeticiones consecutivas de palabras (suave) ------
    max_repeats = 2
    count = 1
    for i in range(1, n_words):
        if words[i].lower() == words[i - 1].lower():
            count += 1
            if count > max_repeats:
                reward -= 1.0
        else:
            count = 1

    # ------ repetición de n-grams (2 y 3) ------
    for n in (2, 3):
        if n_words >= n:
            ngrams = [" ".join(words[i:i+n]).lower() for i in range(n_words - n + 1)]
            counts = Counter(ngrams)
            for seq, c in counts.items():
                if c > 2:
                    reward -= (c - 2) * 0.5 * n  # penalización suave

    # ------ penalizar tokens absurdamente largos al final ------
    if n_words and len(words[-1]) > 20:
        reward -= 3.0

    # ------ penalizar frases con apariencia de "spam"/call-to-action ------
    low = text.lower()
    spam_terms = ["sign up", "click here", "subscribe", "follow us", "buy now", "visit our", "newsletter"]
    for term in spam_terms:
        if term in low:
            reward -= 5.0

    # ------ normalizar rango final para estabilidad en PPO ------
    reward = max(-20.0, min(20.0, reward))

    return float(reward)
