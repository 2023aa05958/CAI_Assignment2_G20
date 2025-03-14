import os
import re
import pickle
import streamlit as st
import pysqlite3 as sqlite3
import chromadb
import json
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tenacity import retry, stop_after_attempt, wait_fixed  

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
print(sqlite3.sqlite_version)

# Paths
CHROMA_DB_DIR = "chroma_db"
DOCUMENTS_DIR = "data"
BM25_MODEL_PATH = "bm25_model.pkl"

# ✅ Load Smaller Language Model (SLM) with Retry Guardrail
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  
def load_slm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # ✅ Choose a smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ Load and Process Documents with Cleaning
def clean_text(text):
    """ Remove special characters, extra spaces, and normalize text """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s.,$%()]', '', text)  # Remove special characters
    return text.strip()

def load_documents(folder_path=DOCUMENTS_DIR):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            file_path = os.path.join(folder_path, file)
            loader = UnstructuredWordDocumentLoader(file_path)
            text = loader.load()[0].page_content  
            text = clean_text(text)
            documents.append(Document(page_content=text, metadata={"source": file}))

    return documents

# ✅ Split documents into structured chunks
def split_documents(documents, chunk_size=600, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents) if documents else []

# ✅ Initialize or Load ChromaDB
def setup_or_load_vector_db(chunks, embedding_model):
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    return Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_DB_DIR)

# ✅ Load or Train BM25 Model
def setup_or_load_bm25(chunks):
    """ Loads BM25 from pickle if available, otherwise trains and saves it """
    if os.path.exists(BM25_MODEL_PATH):
        with open(BM25_MODEL_PATH, "rb") as f:
            bm25_model, corpus = pickle.load(f)
        print("✅ Loaded BM25 model from file.")
    else:
        corpus = [doc.page_content for doc in chunks]
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25_model = BM25Okapi(tokenized_corpus)
        with open(BM25_MODEL_PATH, "wb") as f:
            pickle.dump((bm25_model, corpus), f)
        print("✅ Trained and saved BM25 model.")
    
    return bm25_model, corpus

# ✅ Guardrails: Query Validation using Keywords
def validate_financial_query(query):
    """ Uses a keyword-based filter to check for financial queries """
    financial_keywords = ["revenue", "profit", "loss", "growth", "earnings", "net income", "Microsoft", "fiscal year"]
    if any(keyword.lower() in query.lower() for keyword in financial_keywords):
        return query
    return None  # Rejects irrelevant queries

# ✅ Hybrid Retrieval (BM25 + Embeddings)
def hybrid_financial_retrieval(vector_db, bm25_model, corpus, query, k=3):
    # BM25 retrieval
    query_tokens = query.split()
    bm25_scores = bm25_model.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_results = [(Document(page_content=corpus[i]), bm25_scores[i]) for i in top_bm25_indices]

    # Embedding-based retrieval
    dense_results = vector_db.similarity_search_with_score(query, k=k)
    dense_results = [(Document(page_content=doc.page_content), score) for doc, score in dense_results]

    # ✅ Normalize BM25 scores if applicable
    if bm25_scores.size > 0:
        max_bm25 = max(bm25_scores)
        if max_bm25 > 0:
            bm25_results = [(doc, score / max_bm25) for doc, score in bm25_results]

    # Merge results
    combined_results = bm25_results + dense_results
    unique_results = {}
    for doc, score in combined_results:
        if doc.page_content in unique_results:
            unique_results[doc.page_content].append(score)
        else:
            unique_results[doc.page_content] = [score]

    # Calculate final confidence score (mean of available scores)
    final_results = [(Document(page_content=doc), np.mean(scores)) for doc, scores in unique_results.items()]
    
    # Sort by confidence
    final_results.sort(key=lambda x: x[1], reverse=True)

    return final_results

# ✅ Guardrail: Filter AI Output for Financial Relevance
def filter_financial_ai_output(response):
    """ Prevents AI from generating incorrect or irrelevant financial data. """
    if "I don't know" in response or "I'm not sure" in response:
        return "❌ AI could not find a reliable financial answer. Please refine your query."
    return response

# ✅ Memory-Augmented Retrieval (MAR)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

# ✅ Initialize Financial Variables
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = load_documents()
chunks = split_documents(documents)
vector_db = setup_or_load_vector_db(chunks, embedding_model)  
bm25_model, corpus = setup_or_load_bm25(chunks)
llm = load_slm()

# ✅ Streamlit UI with Financial Guardrails
def main():
    st.set_page_config(page_title="Financial Report Chatbot (SLM)", layout="wide")
    st.title("📊 Financial Report Chatbot (SLM)")
    st.write("Ask questions about **Microsoft's financial performance** in 2023 & 2024.")

    query = st.text_input("🔍 Enter your financial query:")

    if st.button("Search & Generate Response"):
        validated_query = validate_financial_query(query)
        if not validated_query:
            st.warning("❌ Invalid or non-financial query. Please enter a meaningful financial question.")
            return

        with st.spinner("Fetching financial data..."):
            retrieved_docs = hybrid_financial_retrieval(vector_db, bm25_model, corpus, validated_query)

            if not retrieved_docs:
                st.error("❌ No relevant financial documents found. Try reindexing or refining your query.")
                return

            context = "\n\n".join([doc[0].page_content for doc in retrieved_docs])
            confidence_score = np.mean([doc[1] for doc in retrieved_docs])

            prompt = (
                f"Summarize Microsoft's financial performance for 2023 and 2024, "
                f"focusing on revenue, profit, and growth.\n\nContext:\n{context}"
            )

            response = llm(prompt, max_length=512, temperature=0.7)[0]["generated_text"]
            response = filter_financial_ai_output(response)

            st.subheader("🤖 AI-Generated Financial Response:")
            st.write(response)
            st.markdown(f"**🔍 Confidence Score: {confidence_score:.2f}**")

if __name__ == "__main__":
    main()
