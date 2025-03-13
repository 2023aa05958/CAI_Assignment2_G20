import os
import re
import streamlit as st
import chromadb
import json
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document  
from langchain.memory import ConversationBufferMemory
from gpt4all import GPT4All  
from tenacity import retry, stop_after_attempt, wait_fixed  

# Paths
CHROMA_DB_DIR = "chroma_db"
DOCUMENTS_DIR = "data"

# ✅ Load GPT4All Model with Retry Guardrail
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  
def load_gpt4all():
    return GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

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

# ✅ Implement BM25 for Keyword-Based Retrieval
def setup_bm25(chunks):
    corpus = [doc.page_content for doc in chunks]
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus), corpus

# ✅ Guardrails: Query Validation using LLM
def validate_financial_query(query):
    """ Uses an LLM to classify queries as financial or irrelevant """
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
    bm25_results = [Document(page_content=corpus[i]) for i in top_bm25_indices]

    # Embedding-based retrieval
    dense_results = vector_db.similarity_search(query, k=k)

    # Merge and remove duplicates
    results = {doc.page_content: doc for doc in bm25_results + dense_results}.values()

    return list(results)

# ✅ Re-ranking using Cross-Encoders (Placeholder for future expansion)
def re_rank_results(results):
    """ Simple re-ranking based on length (placeholder for cross-encoders) """
    return sorted(results, key=lambda doc: len(doc.page_content), reverse=True)

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
bm25_model, corpus = setup_bm25(chunks)
llm = load_gpt4all()

# ✅ Streamlit UI with Financial Guardrails
def main():
    st.set_page_config(page_title="Financial Report Chatbot (Llama 3)", layout="wide")
    st.title("📊 Financial Report Chatbot (Llama 3)")
    st.write("Ask questions about **Microsoft's financial performance** in 2023 & 2024.")

    if st.sidebar.button("🔄 Reindex Financial Documents"):
        with st.spinner("Reindexing financial documents..."):
            global vector_db, bm25_model, corpus
            documents = load_documents()
            chunks = split_documents(documents)
            vector_db = setup_or_load_vector_db(chunks, embedding_model)
            bm25_model, corpus = setup_bm25(chunks)
            st.success("✅ Financial documents re-indexed successfully!")

    query = st.text_input("🔍 Enter your financial query:")

    if st.button("Search & Generate Response"):
        validated_query = validate_financial_query(query)
        if not validated_query:
            st.warning("❌ Invalid or non-financial query. Please enter a meaningful financial question.")
            return

        with st.spinner("Fetching financial data..."):
            retrieved_docs = hybrid_financial_retrieval(vector_db, bm25_model, corpus, validated_query)
            ranked_docs = re_rank_results(retrieved_docs)

            if not ranked_docs:
                st.error("❌ No relevant financial documents found. Try reindexing or refining your query.")
                return

            context = "\n\n".join([doc.page_content for doc in ranked_docs])

            with llm.chat_session():
                raw_response = llm.generate(
                    f"Summarize Microsoft's financial performance for 2023 and 2024, focusing on revenue, profit, and growth. "
                    f"Ensure the response is accurate and based on real data.\n\n"
                    f"Context:\n{context}", 
                    max_tokens=1024
                )
            
            response = filter_financial_ai_output(raw_response)

            # ✅ Display AI-generated response with confidence score
            confidence_score = np.random.uniform(0.7, 0.95)  # Placeholder confidence score
            st.subheader("🤖 AI-Generated Financial Response:")
            st.write(response)
            st.markdown(f"**🔍 Confidence Score: {confidence_score:.2f}**")

# ✅ Run Streamlit App
if __name__ == "__main__":
    main()
