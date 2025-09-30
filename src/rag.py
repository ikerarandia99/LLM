# rag.py
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import *

# -------------------------------
# Cargar vectorstore
# -------------------------------
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local(
        os.path.join(BASE_DIR, "faiss_index"),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# -------------------------------
# Configurar LLM (usa Flan-T5-small en CPU)
# -------------------------------
def setup_llm(model_name="google/flan-t5-small"):  # 🔑 más ligero para CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # 🔑 CPU
        max_new_tokens=200,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm, tokenizer, llm_pipeline

# -------------------------------
# Crear cadena QA
# -------------------------------
def create_qa_chain(vectorstore, llm, k=2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt_template = """Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""
    prompt = PromptTemplate(input_variables=["query", "context"], template=prompt_template)

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose=True,
    )
    return retrieval_qa

# -------------------------------
# Limpieza de output
# -------------------------------
def clean_generated_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)   # quitar saltos de línea múltiples
    text = re.sub(r' +', ' ', text)    # quitar espacios dobles
    return text.strip()

# -------------------------------
# Consulta limpia (usando contexto)
# -------------------------------
def ask_question_clean(qa_chain, tokenizer, llm_pipeline, question, top_k=3):
    # Recuperar documentos relevantes
    retrieved_docs = qa_chain.retriever.get_relevant_documents(question)[:top_k]
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Construir prompt con contexto
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    result = llm_pipeline(prompt, max_new_tokens=200, do_sample=False)

    if isinstance(result, list) and "generated_text" in result[0]:
        answer = result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        answer = result["generated_text"]
    elif isinstance(result, str):
        answer = result
    else:
        answer = str(result)

    # Limpiar output
    answer = clean_generated_text(answer)

    return answer, retrieved_docs

# -------------------------------
# Función para Streamlit
# -------------------------------
def rag_query(query: str) -> str:
    vectorstore = load_vectorstore()
    llm, tokenizer, hf_pipeline = setup_llm()
    qa_chain = create_qa_chain(vectorstore, llm)

    answer, sources = ask_question_clean(qa_chain, tokenizer, hf_pipeline, query)
    return query, answer

# -------------------------------
# Ejemplo standalone
# -------------------------------
if __name__ == "__main__":
    query = "What is general relativity?"
    output = rag_query(query)
    print("Answer:", output)
