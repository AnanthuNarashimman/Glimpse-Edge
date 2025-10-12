import gradio as gr
import chromadb
import os
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# --- NEW: Import for OCR ---
import easyocr

# --- Force UTF-8 encoding to prevent Unicode errors ---
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- 1. INITIALIZE MODELS AND DATABASE ---
print("--- Initializing Glimpse Offline Backend ---")

# --- LLM for Text Generation ---
MODEL_PATH = "./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
try:
    LLM = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)
    print("Local LLM loaded successfully.")
except Exception as e:
    print(f"Error loading LLM: {e}")
    LLM = None

# --- Model for Embeddings ---
try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Local Embedding Model loaded successfully.")
except Exception as e:
    print(f"Error loading Embedding Model: {e}")
    EMBEDDING_MODEL = None

# --- NEW: Initialize the OCR Reader instead of the captioning model ---
try:
    print("Loading local OCR model... (This may download files on the first run)")
    # This will download the OCR models for English the first time it's run.
    # verbose=False suppresses the Unicode progress bar that causes encoding errors
    OCR_READER = easyocr.Reader(['en'], verbose=False) 
    print("Local OCR Model loaded successfully.")
except Exception as e:
    print(f"Error loading OCR Model: {e}")
    OCR_READER = None

# --- Vector Database ---
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db_offline")
COLLECTION_NAME = "glimpse_offline_brain_multimodal"
COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
print(f"ChromaDB ready. Collection '{COLLECTION_NAME}' has {COLLECTION.count()} items.")

# --- 2. CORE LOGIC FUNCTIONS ---

def _extract_text_from_document(file_path: str) -> str:
    # This function for documents remains the same
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
        print(f"Error extracting text from document {os.path.basename(file_path)}: {e}")
        return ""

# --- MODIFIED: This function now uses OCR to READ text from images ---
def _extract_text_from_image_ocr(image_path: str) -> str:
    """Extracts text from an image file using EasyOCR."""
    if not OCR_READER:
        print("ERROR: OCR model not loaded. Skipping text extraction from image.")
        return ""
    
    print(f"\n{'='*60}")
    print(f"Starting OCR on image: {os.path.basename(image_path)}")
    print(f"Full image path: {image_path}")
    
    try:
        # Verify the image file exists and is accessible
        if not os.path.exists(image_path):
            print(f"ERROR: Image file does not exist: {image_path}")
            return ""
        
        # Check file size
        file_size = os.path.getsize(image_path)
        print(f"Image file size: {file_size} bytes")
        
        if file_size == 0:
            print(f"ERROR: Image file is empty (0 bytes)")
            return ""
        
        # Try to load the image first to verify it's valid
        try:
            from PIL import Image
            img = Image.open(image_path)
            print(f"Image loaded successfully: {img.size} pixels, mode: {img.mode}")
            img.close()
        except Exception as img_err:
            print(f"ERROR: Cannot open image file: {img_err}")
            return ""
        
        print("Reading image with EasyOCR (this may take a moment)...")
        
        # Try with default settings first
        result = OCR_READER.readtext(image_path, detail=1)  # detail=1 gives us confidence scores
        
        print(f"OCR raw result count: {len(result)}")
        
        if not result or len(result) == 0:
            print(f"WARNING: OCR found no text in {os.path.basename(image_path)}")
            print("This could mean:")
            print("  1. The image contains no text")
            print("  2. The text is too small or blurry")
            print("  3. The text is in a language not supported")
            print("  4. The image quality is too low")
            return ""
        
        # Extract text and confidence scores
        extracted_texts = []
        for detection in result:
            bbox, text, confidence = detection
            print(f"  Detected: '{text}' (confidence: {confidence:.2f})")
            # Only include text with reasonable confidence (>0.1)
            if confidence > 0.1:
                extracted_texts.append(text)
        
        if not extracted_texts:
            print(f"WARNING: All detected text had too low confidence")
            return ""
        
        # Join all text pieces
        extracted_text = " ".join(extracted_texts)
        
        print(f"\nOCR SUCCESS:")
        print(f"  - Extracted {len(extracted_texts)} text regions")
        print(f"  - Total characters: {len(extracted_text)}")
        print(f"  - Full text: {extracted_text}")
        print(f"{'='*60}\n")
        
        return extracted_text
        
    except Exception as e:
        print(f"ERROR processing image with OCR {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()
        return ""

# --- MODIFIED: The main ingestion function is now a smarter "traffic cop" ---
def ingest_files(files):
    if not files:
        yield "No files uploaded. Please select files to process."
        return
    
    total_files = len(files)
    print(f"--- Starting ingestion for {total_files} files ---")
    
    doc_extensions = ['.pdf', '.docx', '.txt']
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    
    for i, file in enumerate(files):
        status_message = f"Processing file {i + 1} of {total_files}: {os.path.basename(file.name)}..."
        print(status_message)
        yield status_message
        
        # Gradio's file object has the path in file.name
        file_path = file.name
        print(f"DEBUG: Full file path from Gradio: {file_path}")
        print(f"DEBUG: File exists check: {os.path.exists(file_path)}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        full_text = ""
        
        print(f"File extension detected: {file_ext}")
        
        if file_ext in doc_extensions:
            print(f"Processing as DOCUMENT: {os.path.basename(file_path)}")
            full_text = _extract_text_from_document(file_path)
        elif file_ext in image_extensions:
            print(f"Processing as IMAGE with OCR: {os.path.basename(file_path)}")
            # --- THE FIX: Call the new OCR function for images ---
            full_text = _extract_text_from_image_ocr(file_path)
        else:
            print(f"Skipping '{os.path.basename(file_path)}'. Unsupported file type.")
            continue
        
        print(f"Extracted text length: {len(full_text)} characters")
        print(f"Text preview: {full_text[:100] if full_text else 'EMPTY'}...")
        
        if not full_text or not full_text.strip():
            print(f"ERROR: Skipping '{os.path.basename(file_path)}' due to empty content.")
            continue
        
        print(f"Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        print(f"Created {len(chunks)} chunks from the text")
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = EMBEDDING_MODEL.encode(chunks).tolist()
        ids = [f"{os.path.basename(file.name)}_{j}" for j in range(len(chunks))]
        metadatas = [{"source": os.path.basename(file.name)} for _ in chunks]
        
        print(f"Adding {len(chunks)} chunks to the knowledge base...")
        COLLECTION.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"Successfully added {len(chunks)} chunks from {os.path.basename(file.name)}")
    
    final_message = f"Successfully processed {total_files} files. The Brain now contains a total of {COLLECTION.count()} knowledge chunks."
    print(final_message)
    yield final_message

# The chat function remains the same
def chat_with_docs(user_message, chat_history):
    if COLLECTION.count() == 0:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "The knowledge base is empty. Please upload some files first."})
        return chat_history
    
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})

    query_embedding = EMBEDDING_MODEL.encode(user_message).tolist()
    results = COLLECTION.query(query_embeddings=[query_embedding], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        chat_history[-1]["content"] = "I could not find any relevant information to answer that question."
        yield chat_history
        return
        
    context = "\n\n".join(results['documents'][0])
    prompt = f"Use the following context to answer the user's question. Context:\n{context}\n\nUser Question:\n{user_message}\n\nAnswer:"
    
    output = LLM(prompt, max_tokens=512, stop=["User Question:"], stream=True)
    for chunk in output:
        chat_history[-1]["content"] += chunk['choices'][0]['text']
        yield chat_history

# --- 3. BUILD THE GRADIO UI ---
with gr.Blocks(title="Glimpse", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# Glimpse: Your Offline Intelligence Engine")
    
    with gr.Tab("Ingest Knowledge"):
        gr.Markdown("Upload your documents (.pdf, .docx, .txt) and images (.png, .jpg).")
        file_input = gr.File(
            label="Upload Files", 
            file_count="multiple", 
            file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
        )
        ingest_button = gr.Button("Process Files", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("Chat with your Brain"):
        gr.Markdown("Ask questions about the files you've uploaded.")
        chatbot = gr.Chatbot(label="Conversation", type="messages")
        msg_box = gr.Textbox(label="Your Question", placeholder="Ask about the content of your files...")
        clear_button = gr.ClearButton([msg_box, chatbot])

    ingest_button.click(ingest_files, inputs=[file_input], outputs=[ingest_status])
    msg_box.submit(chat_with_docs, inputs=[msg_box, chatbot], outputs=[chatbot])

# --- 4. LAUNCH THE APP ---
if __name__ == "__main__":
    print("--- Launching Gradio UI ---")
    demo.launch(share=False, server_port=7860)