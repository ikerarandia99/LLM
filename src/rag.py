from transformers import pipeline, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from config import *

# -------------------------------
# Cargar vectorstore
# -------------------------------
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        os.path.join(BASE_DIR, "faiss_index"),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# -------------------------------
# Configurar LLM (usa el modelo entrenado RL)
# -------------------------------
def setup_llm():
    model_name = "gpt2"  # o tu RL-trained model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device=-1,          # CPU (-1), o GPU si quieres
        max_new_tokens=200,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm, tokenizer, llm_pipeline

# -------------------------------
# Crear cadena QA
# -------------------------------
def create_qa_chain(vectorstore, llm, k=2):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt_template = """Answer the following question based on the provided context:
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
# Truncado seguro
# -------------------------------
def truncate_to_max_tokens(text, tokenizer, max_tokens=800):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)

# -------------------------------
# Limpieza de output
# -------------------------------
def clean_generated_text(text: str) -> str:
    # quitar saltos de línea múltiples
    text = re.sub(r'\n+', ' ', text)
    # quitar espacios dobles
    text = re.sub(r' +', ' ', text)
    # eliminar repeticiones simples de “Answer:”
    text = re.sub(r'(Answer:\s*)+', 'Answer: ', text)
    return text.strip()

# -------------------------------
# Consulta limpia
# -------------------------------
def ask_question_clean(qa_chain, tokenizer, llm_pipeline, question, top_k=3, max_tokens=800):
    # Recuperar documentos (solo para internamente, no se muestra)
    retrieved_docs = qa_chain.retriever.get_relevant_documents(question)[:top_k]

    # Prompt instruccional para respuestas concisas y coherentes
    prompt = f"""
Answer the following question concisely and clearly. Avoid including the retrieved context.
If relevant, continue answering related sub-questions. Provide coherent sentences.
Question: {question}
Answer:"""

    result = llm_pipeline(prompt, max_new_tokens=200, do_sample=True)

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
    return answer

# -------------------------------
# Ejemplo standalone
# -------------------------------
if __name__ == "__main__":
    query = "What is general relativity?"
    output = rag_query(query)
    print("Answer:", output)
