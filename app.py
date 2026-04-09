import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pickle
import json
import glob

# Define directory for storing files
DATA_DIR = "rudrawarparth/Ask_from_Web"  # Adjust if "hugging face files" is a different path
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
JSON_FILE_PATH = os.path.join(DATA_DIR, "data.json")
FAISS_FILE_PATH = os.path.join(DATA_DIR, "dat_embeding.pkl")
MAPPING_FILE_PATH = os.path.join(DATA_DIR, "url_mapping.json")

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'url_mapping' not in st.session_state:
    st.session_state.url_mapping = {}

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("🔍 Smart Research Assistant")

# Function to read URLs from url.txt
def read_urls_from_file(file_path="url.txt"):
    try:
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
        return urls
    except FileNotFoundError:
        st.error(f"❌ File {file_path} not found.")
        return []
    except Exception as e:
        st.error(f"❌ Error reading {file_path}: {str(e)}")
        return []

# Function to load or create FAISS database
def load_or_create_db():
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Check if FAISS database exists on disk
    if os.path.exists(FAISS_FILE_PATH):
        try:
            with open(FAISS_FILE_PATH, 'rb') as f:
                st.session_state.vector_db = pickle.load(f)
            st.success(f"✅ Loaded existing FAISS database from {FAISS_FILE_PATH}")
            # Load URL mapping if available
            if os.path.exists(MAPPING_FILE_PATH):
                with open(MAPPING_FILE_PATH, 'r') as f:
                    st.session_state.url_mapping = json.load(f)
            return st.session_state.vector_db
        except Exception as e:
            st.error(f"❌ Error loading FAISS database: {str(e)}")
    
    # Create new database if none exists
    if st.session_state.vector_db is None:
        st.session_state.vector_db = FAISS.from_texts([""], embedding)
        st.session_state.vector_db.delete([st.session_state.vector_db.index_to_docstore_id[0]])
    return st.session_state.vector_db

def add_urls_to_db(urls):
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = load_or_create_db()
    
    # Filter out empty or duplicate URLs
    existing_urls = set(st.session_state.url_mapping.values())
    urls = [url for url in urls if url and url not in existing_urls]
    if not urls:
        st.error("❌ No valid URLs to process.")
        return 0
    
    # Assign new URL numbers
    current_max = max(st.session_state.url_mapping.keys(), default=0)
    new_mapping = {i + current_max + 1: url for i, url in enumerate(urls)}
    st.session_state.url_mapping.update(new_mapping)
    
    total_added = 0
    json_data = []
    try:
        # Step 1: Batch load all URLs
        loader = WebBaseLoader(urls)
        data = loader.load()
        
        # Step 2: Save scraped data to JSON
        for doc in data:
            url = doc.metadata.get('source', '')
            url_num = next((num for num, u in new_mapping.items() if u == url), None)
            if url_num is None:
                st.warning(f"⚠️ Skipping {url[:50]}: No matching URL number.")
                continue
            json_entry = {
                "url_number": url_num,
                "url": url,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            json_data.append(json_entry)
            # Pop-up confirmation for each URL
            st.toast(f"✅ Link scraped and saved to {JSON_FILE_PATH}: {url[:50]}...", icon="✅")
        
        # Write all scraped data to JSON file
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        st.success(f"✅ All scraped data saved to {JSON_FILE_PATH}")
        
        # Step 3: Process JSON data for vector embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        all_chunks = []
        for item in json_data:
            doc = Document(page_content=item['content'], metadata={'source': item['url'], 'url_number': item['url_number']})
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        if all_chunks:
            # Create FAISS database
            temp_db = FAISS.from_documents(all_chunks, embedding)
            db.merge_from(temp_db)
            total_added = len(all_chunks)
            st.success(f"✅ Processed {total_added} chunks into vector database.")
        
        # Save FAISS database and URL mapping
        with open(FAISS_FILE_PATH, 'wb') as f:
            pickle.dump(db, f)
        with open(MAPPING_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.url_mapping, f, ensure_ascii=False, indent=2)
        st.success(f"✅ Saved FAISS database to {FAISS_FILE_PATH}")
        
    except Exception as e:
        st.error(f"❌ Error processing URLs: {str(e)}")
    
    return total_added

def prompt_helper():
    template = """You are a research assistant. Provide detailed, structured answers with proper attribution.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer Format Guidelines:
    1. Start with a brief overview (1-2 sentences)
    2. Break down into numbered steps for processes
    3. Use bullet points for key facts
    4. Include all relevant details
    5. Conclude with complete source attribution
    
    Structure your response like this example:
    
    ### Overview
    [Concise summary of the answer]
    
    ### Detailed Explanation
    1. First key point or step...
    2. Second key point or step...
    3. Third key point or step...
    
    - Additional important detail...
    - Another relevant fact...
    
    ### References
    This information was gathered from the following sources:
    [1] Complete Source URL 1
    [2] Complete Source URL 2
    
    For more detailed information, please visit:
    [Most Relevant Source URL]
    """
    return ChatPromptTemplate.from_template(template)

def get_answer(question):
    if st.session_state.vector_db is None or len(st.session_state.vector_db.docstore._dict) == 0:
        return {
            "answer": "### ⚠️ Knowledge Base Empty\nPlease add some sources first.",
            "sources": []
        }
    
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
    
    def format_docs(docs):
        formatted = []
        sources = set()
        for doc in docs:
            formatted.append(f"Content: {doc.page_content}\nSource: [doc.metadata.get('url_number', '?')]")
            if 'url_number' in doc.metadata:
                sources.add(doc.metadata['url_number'])
        return "\n\n".join(formatted), sorted(sources)
    
    prompt = prompt_helper()
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0.2,
        top_k=40,
        top_p=0.9
    )
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        response = chain.invoke(question)
        _, source_nums = format_docs(retriever.get_relevant_documents(question))
        return {
            "answer": response,
            "sources": source_nums
        }
    except Exception as e:
        return {
            "answer": f"### ⚠️ Error\nCould not generate answer: {str(e)}",
            "sources": []
        }

# Automatically load URLs from url.txt on startup if no knowledge base exists
if st.session_state.vector_db is None or len(st.session_state.vector_db.docstore._dict) == 0:
    urls = read_urls_from_file()
    if urls:
        with st.spinner("Processing URLs from url.txt..."):
            added = add_urls_to_db(urls)
            if added > 0:
                st.success(f"Added {added} content chunks from {len(urls)} sources")
            else:
                st.warning("No valid URLs processed from url.txt.")

# Main question area
col1, col2 = st.columns([3, 1])
with col1:
    user_question = st.text_input("Ask your research question:", placeholder="Type your question here...")
with col2:
    query_button = st.button("🚀 Get Answer")

# Handle queries
if query_button and user_question:
    if st.session_state.vector_db is None or len(st.session_state.vector_db.docstore._dict) == 0:
        st.warning("Please ensure valid sources are available in url.txt.")
    else:
        with st.spinner("Researching your question..."):
            result = get_answer(user_question)
            
            st.markdown("## 📚 Research Results")
            st.markdown(result["answer"])
            
            if result["sources"]:
                st.markdown("---")
                st.markdown("### 🔍 Source Attribution")
                for num in result["sources"]:
                    if num in st.session_state.url_mapping:
                        url = st.session_state.url_mapping[num]
                        st.markdown(
                            f"**Source [{num}]**: {url}\n\n"
                            f"For more detailed information, visit: [{url}]({url})"
                        )
            
            st.markdown("### 📊 Knowledge Base Stats")
            st.metric("Total Content Chunks", len(st.session_state.vector_db.docstore._dict))
            st.metric("Total Sources", len(st.session_state.url_mapping))