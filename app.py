import streamlit as st
import sys
import os
from pathlib import Path
from src.config import *

# Adjust sys.path to import from src
sys.path.append(os.path.join(BASE_DIR, "src"))

st.set_page_config(page_title="LLM Inference", layout="centered")
st.title("LLM Inference Comparisons")

# Imports for generation
from generate import generate_text
from rag import rag_query

# ---------------------------
# Section 1: Simple sentences
# ---------------------------
st.markdown("### Prompt for simple sentence comparison (distilgpt2 vs RL-trained)")
prompt_simple = st.text_area("Enter a prompt for simple sentences:", height=100, key="simple_prompt")

if st.button("Generate (RL)"):
    if not prompt_simple.strip():
        st.warning("Please enter a prompt for simple sentences.")
    else:
        with st.spinner("Generating outputs..."):
            st.markdown("## Comparison 1: Base 'distilgpt2' vs RL-trained model (Simple sentences)")
            st.markdown("This comparison is intended to see how the RL-trained model can complete sentences more naturally.")

            # Base distilgpt2
            base_output = generate_text(prompt_simple, model_name_or_path="distilgpt2", max_new_tokens=35)
            st.markdown("**Base distilgpt2 output:**")
            st.write(base_output)

            # RL-trained model
            rl_model_path = os.path.join(MODEL_DIR, "gpt2-rl/final/")
            rl_output = generate_text(prompt_simple, model_name_or_path=rl_model_path, max_new_tokens=35, device=0)
            st.markdown("**RL-trained model output:**")
            st.write(rl_output)

st.markdown("---")

# ---------------------------
# Section 2: Thematic QA
# ---------------------------
st.markdown("### Prompt for thematic question comparison (gpt2 vs RAG)")
prompt_thematic = st.text_area("Enter a prompt for thematic questions:", height=100, key="thematic_prompt")

if st.button("Generate (RAG)"):
    if not prompt_thematic.strip():
        st.warning("Please enter a prompt for thematic questions.")
    else:
        with st.spinner("Generating outputs..."):
            st.markdown("## Comparison 2: Base 'gpt2' vs RAG (Thematic questions)")
            st.markdown("This comparison is intended to see how RAG improves answers by using document retrieval.")

            # Base gpt2
            gpt2_output = generate_text(prompt_thematic, model_name_or_path="gpt2", max_new_tokens=50)
            st.markdown("**Base gpt2 output:**")
            st.write(gpt2_output)

            # RAG (RL-trained model + retrieval)
            rag_output = rag_query(prompt_thematic)
            st.markdown("**RAG output (RL-trained model + retrieval):**")
            st.write(rag_output[0]+'\n'+rag_output[1])
