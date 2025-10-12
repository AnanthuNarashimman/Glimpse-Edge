import gradio as gr
import chromadb
import os
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# --- 1. INITIALIZE MODELS AND DATABASE ---
print("--- Initializing Glimpse Offline Backend ---")

MODEL_PATH = "./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
try:
    LLM = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)
    print("Local LLM loaded successfully.")
except Exception as e:
    print(f"Error loading LLM: {e}")
    LLM = None

try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Local Embedding Model loaded successfully.")
except Exception as e:
    print(f"Error loading Embedding Model: {e}")
    EMBEDDING_MODEL = None

CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db_offline")
COLLECTION_NAME = "glimpse_offline_brain"
COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
print(f"ChromaDB ready. Collection '{COLLECTION_NAME}' has {COLLECTION.count()} items.")

# --- 2. CORE LOGIC FUNCTIONS ---

def _extract_text_from_file(file_path: str) -> str:
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return text
    except Exception as e:
        print(f"Error extracting text from {os.path.basename(file_path)}: {e}")
        return ""

def ingest_files(files):
    if not files:
        yield "No files uploaded. Please select files to process."
        return

    if not EMBEDDING_MODEL:
        yield "Error: Embedding Model not loaded. Cannot process files."
        return
    
    total_files = len(files)
    print(f"--- Starting ingestion for {total_files} files ---")
    
    for i, file in enumerate(files):
        status_message = f"Processing file {i + 1} of {total_files}: {os.path.basename(file.name)}..."
        print(status_message)
        yield status_message
        
        full_text = _extract_text_from_file(file.name)
        
        if not full_text.strip():
            print(f"Skipping '{os.path.basename(file.name)}' due to empty content.")
            continue
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        
        embeddings = EMBEDDING_MODEL.encode(chunks).tolist()
        ids = [f"{os.path.basename(file.name)}_{j}" for j in range(len(chunks))]
        metadatas = [{"source": os.path.basename(file.name)} for _ in chunks]
        
        COLLECTION.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    
    final_message = f"Successfully processed {total_files} files. The Brain now contains a total of {COLLECTION.count()} knowledge chunks."
    print(final_message)
    yield final_message

def chat_with_docs(user_message, chat_history):
    if COLLECTION.count() == 0:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "The knowledge base is empty. Please upload some documents first."})
        return chat_history
    if not LLM or not EMBEDDING_MODEL:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "An AI model failed to load. Please check the console for errors."})
        return chat_history

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})

    query_embedding = EMBEDDING_MODEL.encode(user_message).tolist()
    results = COLLECTION.query(query_embeddings=[query_embedding], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        chat_history[-1]["content"] = "I could not find any relevant information in your documents to answer that question."
        yield chat_history
        return
        
    context = "\n\n".join(results['documents'][0])
    
    prompt = f"""Use the following context to answer the user's question. If the context is not sufficient, say so.

Context:
{context}

User Question:
{user_message}

Answer:
"""
    
    # --- FIX: Removed '\n' from stop sequence and increased max_tokens ---
    output = LLM(
        prompt, 
        max_tokens=512, 
        stop=["User Question:"], # This is a safer stop sequence
        stream=True
    )
    
    for chunk in output:
        chat_history[-1]["content"] += chunk['choices'][0]['text']
        yield chat_history

# --- 3. BUILD THE GRADIO UI ---
with gr.Blocks(title="Glimpse", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# Glimpse: Your Offline Intelligence Engine")
    
    with gr.Tab("Ingest Knowledge"):
        gr.Markdown("Upload your documents (.pdf, .docx, .txt) to build your knowledge base.")
        file_input = gr.File(label="Upload Documents", file_count="multiple", file_types=[".pdf", ".docx", ".txt"])
        ingest_button = gr.Button("Process Files", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("Chat with your Brain"):
        gr.Markdown("Ask questions about the documents you've uploaded.")
        chatbot = gr.Chatbot(label="Conversation", type="messages")
        msg_box = gr.Textbox(label="Your Question", placeholder="Ask about the content of your documents...")
        clear_button = gr.ClearButton([msg_box, chatbot])

    # --- 4. WIRE UP THE UI COMPONENTS ---
    ingest_button.click(ingest_files, inputs=[file_input], outputs=[ingest_status])
    msg_box.submit(chat_with_docs, inputs=[msg_box, chatbot], outputs=[chatbot])

# --- 5. LAUNCH THE APP ---
if __name__ == "__main__":
    print("--- Launching Gradio UI ---")
    demo.launch(share=False, server_port=7860)

