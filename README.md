# LLM - Fine-tuning y RL para Modelos de Lenguaje

Este proyecto explora el fine-tuning y entrenamiento por refuerzo (RL) de modelos de lenguaje tipo GPT-2, así como la integración de técnicas de Retrieval-Augmented Generation (RAG) para mejorar respuestas temáticas. Incluye scripts para entrenamiento, inferencia y una app interactiva con Streamlit.

## Estructura del Proyecto

```
LLM/
├── app.py                # Interfaz Streamlit para comparación de modelos
├── environment.yaml      # Entorno Conda con dependencias
├── src/
│   ├── config.py         # Configuración de rutas y parámetros
│   ├── dataset.py        # Utilidades para datasets
│   ├── generate.py       # Inferencia con modelos base o RL
│   ├── inference.py      # Comparación de outputs (base vs RL)
│   ├── reward.py         # Función de reward para PPO
│   ├── rl_train.py       # Entrenamiento RL (PPO)
│   ├── rag.py            # Retrieval-Augmented Generation (RAG)
│   └── ...
├── models/
│   └── gpt2-rl/
│       └── final/        # Modelo entrenado con RL
└── README.md
```

## Instalación

1. **Clona el repositorio** y entra a la carpeta:

   ```bash
   git clone <repo_url>
   cd LLM
   ```

2. **Crea el entorno Conda**:

   ```bash
   conda env create -f environment.yaml
   conda activate LLM
   ```

3. **Instala dependencias adicionales** (si usas RAG):

   ```bash
   pip install langchain faiss-cpu sentence-transformers
   ```

## Entrenamiento

### Fine-tuning SFT (Supervised)

Utiliza `src/train.py` para entrenar el modelo base (no incluido explícitamente aquí, pero puedes adaptar el script).

### Entrenamiento por Refuerzo (RL, PPO)

Entrena un modelo GPT-2 con PPO usando `src/rl_train.py`:

```bash
python src/rl_train.py
```

- El reward se define en `src/reward.py` y favorece frases bien formadas, sin repeticiones ni finales anómalos.
- El modelo entrenado se guarda en `models/gpt2-rl/final/`.

## Inferencia

### Usar el modelo base o RL

Puedes generar texto con:

- `src/generate.py`: para el modelo base o cualquier modelo HuggingFace.
- `src/inference.py`: compara outputs entre el modelo base y el modelo RL.

Ejemplo:

```bash
python src/generate.py
```

## Retrieval-Augmented Generation (RAG)

El script `src/rag.py` permite responder preguntas usando recuperación de documentos y generación con el modelo RL o base. Requiere un índice FAISS y embeddings (ver función `load_vectorstore`).

## App Interactiva (Streamlit)

Lanza la app para comparar modelos y probar prompts:

```bash
streamlit run app.py
```

- Permite comparar el modelo base (`distilgpt2` o `gpt2`) vs el modelo RL.
- Incluye comparación con RAG para preguntas temáticas (requiere índice FAISS y dependencias de LangChain).

## Configuración

Edita `src/config.py` para ajustar rutas, modelo base y parámetros de entrenamiento.

## Notas

- Los modelos y checkpoints se guardan en `models/gpt2-rl/`.
- Los datos de entrenamiento deben colocarse en la carpeta `data/`.
- El reward para RL está diseñado para premiar frases naturales y penalizar repeticiones o finales incorrectos.

---

¿Quieres agregar instrucciones para crear el índice FAISS, ejemplos de prompts, o detalles sobre datasets?
