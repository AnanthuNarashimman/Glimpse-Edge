import gradio as gr
import chromadb
import os
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sqlite3
import re
import vosk
import json
import wave
import tempfile
import subprocess

# --- 1. INITIALIZE DATABASES & MODELS ---
print("--- Initializing Glimpse Offline Backend ---")

# --- SQLite Database for Brain Management ---
DB_FILE = "brains.db"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS brains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        bio TEXT
    )
""")
conn.commit()
conn.close()
print("Local SQLite DB for Brains initialized.")

# --- AI Models (unchanged) ---
MODEL_PATH = "./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
try:
    LLM = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)
    print("Local LLM loaded.")
except Exception as e:
    LLM = None
try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Local Embedding Model loaded.")
except Exception:
    EMBEDDING_MODEL = None
try:
    IMAGE_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    IMAGE_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Local Image Captioning Model loaded.")
except Exception:
    IMAGE_MODEL = None

# --- Vosk Speech Recognition ---
VOSK_MODEL = None
try:
    VOSK_MODEL_PATH = "./vosk-model-en-us-0.22-lgraph"
    if os.path.exists(VOSK_MODEL_PATH):
        VOSK_MODEL = vosk.Model(VOSK_MODEL_PATH)
        print("Vosk Speech Recognition Model loaded.")
    else:
        print("Vosk model not found. Voice input will be disabled.")
except Exception as e:
    print(f"Failed to load Vosk model: {e}")
    VOSK_MODEL = None

# --- Vector Database ---
CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_db_offline")
print("ChromaDB client ready.")

# --- 2. CORE LOGIC & HELPER FUNCTIONS ---

def get_brain_list():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, bio FROM brains ORDER BY name")
    brains = cursor.fetchall()
    conn.close()
    return brains

def create_brain(name, bio):
    if not name or not name.strip():
        return "Error: Brain name cannot be empty.", get_brain_list()
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', name).lower()
    if not sanitized_name:
        return "Error: Invalid brain name.", get_brain_list()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO brains (name, bio) VALUES (?, ?)", (sanitized_name, bio))
        conn.commit()
        CHROMA_CLIENT.get_or_create_collection(name=sanitized_name)
        message = f"Brain '{sanitized_name}' created."
    except sqlite3.IntegrityError:
        message = f"Error: Brain '{sanitized_name}' already exists."
    finally:
        conn.close()
    return message, get_brain_list()

def delete_brain(brain_name):
    if not brain_name:
        return "Error: No brain selected to delete.", get_brain_list()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM brains WHERE name = ?", (brain_name,))
        conn.commit()
        CHROMA_CLIENT.delete_collection(name=brain_name)
        message = f"Brain '{brain_name}' deleted."
    except Exception as e:
        message = f"Error deleting brain: {e}"
    finally:
        conn.close()
    return message, get_brain_list()

def get_files_in_brain(brain_name):
    """Gets a list of unique source filenames from a ChromaDB collection."""
    if not brain_name: return "No Brain selected."
    try:
        collection = CHROMA_CLIENT.get_collection(name=brain_name)
        metadata = collection.get(include=["metadatas"])['metadatas']
        if not metadata: return "This Brain is empty."
        unique_sources = sorted(list(set(item['source'] for item in metadata)))
        return "\n".join(f"- {source}" for source in unique_sources)
    except Exception:
        return f"Brain '{brain_name}' not found or is empty."

# --- File Processing (unchanged) ---
def _extract_text_from_document(file_path):
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages: text += page.extract_text() or ""
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs: text += para.text + "\n"
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
        return text
    except Exception: return ""

def _get_image_caption_offline(image_path):
    if not IMAGE_MODEL: return ""
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = IMAGE_PROCESSOR(raw_image, return_tensors="pt")
        out = IMAGE_MODEL.generate(**inputs, max_new_tokens=50)
        return IMAGE_PROCESSOR.decode(out[0], skip_special_tokens=True)
    except Exception: return ""

def ingest_files(files, brain_name, progress=gr.Progress()):
    if not brain_name: return "Please select a Brain first.", ""
    if not files: return "No files uploaded.", get_files_in_brain(brain_name)

    collection = CHROMA_CLIENT.get_collection(name=brain_name)
    for i, file in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing: {os.path.basename(file.name)}")
        file_ext = os.path.splitext(file.name)[1].lower()
        full_text = ""
        if file_ext in ['.pdf', '.docx', '.txt']:
            full_text = _extract_text_from_document(file.name)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            full_text = _get_image_caption_offline(file.name)
        
        if not full_text.strip(): continue
        
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(full_text)
        embeddings = EMBEDDING_MODEL.encode(chunks).tolist()
        ids = [f"{os.path.basename(file.name)}_{j}" for j in range(len(chunks))]
        metadatas = [{"source": os.path.basename(file.name)} for _ in chunks]
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    
    status = f"Processed {len(files)} files. Brain '{brain_name}' now has {collection.count()} chunks."
    return status, get_files_in_brain(brain_name)

def chat_with_brain(user_message, chat_history, brain_name):
    if not brain_name:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "Please select a Brain to chat with from the 'Manage Brains' page."})
        return chat_history
        
    collection = CHROMA_CLIENT.get_collection(name=brain_name)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    
    # Query the vector database with metadata
    results = collection.query(
        query_embeddings=[EMBEDDING_MODEL.encode(user_message).tolist()], 
        n_results=3,
        include=["documents", "metadatas"]
    )
    
    # Check if we have any relevant documents
    if not results.get('documents') or not results['documents'][0] or collection.count() == 0:
        # No documents in the brain or no relevant chunks found
        chat_history[-1]["content"] = "❌ No information found in this brain. Please upload relevant documents first."
        yield chat_history
        return
    
    # Extract context and sources
    context = "\n\n".join(results['documents'][0])
    sources = []
    if results.get('metadatas') and results['metadatas'][0]:
        seen_sources = set()
        for metadata in results['metadatas'][0]:
            source = metadata.get('source', 'Unknown')
            if source not in seen_sources:
                sources.append(source)
                seen_sources.add(source)
    
    prompt = f"Context:\n{context}\n\nQuestion:\n{user_message}\n\nAnswer:"
    output = LLM(prompt, max_tokens=512, stop=["Question:"], stream=True)
    
    # Stream the response
    for chunk in output:
        chat_history[-1]["content"] += chunk['choices'][0]['text']
        yield chat_history
    
    # Add citations after the response
    if sources:
        citation_text = "\n\n---\n**📚 Sources:**\n" + "\n".join([f"- {source}" for source in sources])
        chat_history[-1]["content"] += citation_text
        yield chat_history

def transcribe_audio(audio_path):
    """
    Convert audio file to text using Vosk with automatic format conversion.
    """
    if not VOSK_MODEL:
        return "❌ Voice recognition is not available. Vosk model not loaded."
    
    if not audio_path:
        return ""
    
    try:
        import subprocess
        import os
        
        # Create a temporary file for converted audio
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        try:
            # Convert audio to the required format using ffmpeg
            # Gradio typically saves audio, but we need to ensure it's in the right format
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-ar', '16000',  # 16000 Hz sample rate
                '-ac', '1',      # Mono
                '-sample_fmt', 's16',  # 16-bit
                temp_wav_path
            ], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            
            audio_to_process = temp_wav_path
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # FFmpeg not available, try to use the original file
            audio_to_process = audio_path
        
        # Open the audio file
        wf = wave.open(audio_to_process, "rb")
        
        # Check if format is acceptable
        if wf.getnchannels() != 1:
            wf.close()
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            return "❌ Audio must be mono. Please check your microphone settings."
        
        if wf.getframerate() not in [8000, 16000, 32000, 48000]:
            wf.close()
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            return f"❌ Sample rate {wf.getframerate()} Hz not supported. Need 8000, 16000, 32000, or 48000 Hz."
        
        # Create recognizer
        rec = vosk.KaldiRecognizer(VOSK_MODEL, wf.getframerate())
        rec.SetWords(True)
        
        # Process audio
        transcribed_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcribed_text += result.get("text", "") + " "
        
        # Get final result
        final_result = json.loads(rec.FinalResult())
        transcribed_text += final_result.get("text", "")
        
        wf.close()
        
        # Clean up temp file
        if os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except:
                pass
        
        return transcribed_text.strip() if transcribed_text.strip() else "❌ No speech detected. Please try again and speak clearly."
        
    except Exception as e:
        return f"❌ Error processing audio: {str(e)}"

# --- 3. BUILD THE GRADIO UI (TWO PAGES ONLY) ---
with gr.Blocks(title="Glimpse", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# Glimpse: Your Offline Intelligence Engine")
    
    brain_list_state = gr.State(get_brain_list())

    with gr.Tabs():
        # === PAGE 1: BRAIN MANAGEMENT (NO REDIRECTS) ===
        with gr.Tab("🧠 Manage Brains"):
            gr.Markdown("## Create and Manage Your Knowledge Brains")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Create New Brain")
                    new_brain_name = gr.Textbox(label="Brain Name", placeholder="e.g., project-alpha")
                    new_brain_bio = gr.Textbox(label="Description", placeholder="Brief description of this brain")
                    create_brain_button = gr.Button("Create Brain", variant="primary")
                    manage_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### Your Brains")
                    brain_cards = gr.Dataset(
                        components=[gr.Textbox(visible=False), gr.Textbox(visible=False)],
                        headers=["Name", "Description"],
                        samples=brain_list_state.value,
                        label="All Brains"
                    )
                    with gr.Row():
                        selected_brain_to_delete = gr.Textbox(label="Selected Brain", interactive=False)
                        delete_brain_button = gr.Button("Delete Selected Brain", variant="stop")

        # === PAGE 2: CHAT PAGE ===
        with gr.Tab("💬 Chat"):
            gr.Markdown("## Chat with Your Brain")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Documents")
                    
                    # Brain selector dropdown
                    brain_selector = gr.Dropdown(
                        label="Select Brain",
                        choices=[b[0] for b in brain_list_state.value],
                        interactive=True
                    )
                    
                    # File upload area
                    file_input = gr.File(
                        label="Upload Files to Current Brain", 
                        file_count="multiple", 
                        file_types=[".pdf", ".docx", ".txt", ".png", ".jpg"]
                    )
                    ingest_button = gr.Button("Add to Brain", variant="primary")
                    ingest_status = gr.Textbox(label="Upload Status", interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("### Files in Brain")
                    file_list = gr.Markdown("Select a brain to see files")

                with gr.Column(scale=2):
                    gr.Markdown("### Conversation")
                    chatbot = gr.Chatbot(type="messages", height=500)
                    
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about the content in this brain...", 
                        show_label=False,
                        lines=1
                    )
                    
                    with gr.Row():
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="🎤 Voice Input",
                            show_label=True
                        )
                        clear_button = gr.ClearButton([msg_box, chatbot], value="Clear Chat")

    # --- 4. WIRE UP THE LOGIC ---

    # Task 1: Create Brain and clear input fields
    def create_brain_and_clear(name, bio):
        status, updated_list = create_brain(name, bio)
        brain_names = [b[0] for b in updated_list]
        return (
            status,                                  # manage_status
            updated_list,                            # brain_list_state
            "",                                      # clear name field
            "",                                      # clear bio field
            gr.update(samples=updated_list),         # update brain_cards
            gr.update(choices=brain_names)           # update brain_selector dropdown
        )
    
    create_brain_button.click(
        fn=create_brain_and_clear,
        inputs=[new_brain_name, new_brain_bio],
        outputs=[manage_status, brain_list_state, new_brain_name, new_brain_bio, brain_cards, brain_selector]
    )
    
    # Update brain dropdown and cards when brain list changes (for delete operations)
    brain_list_state.change(
        fn=lambda brains: (
            gr.update(samples=brains),
            gr.update(choices=[b[0] for b in brains])
        ), 
        inputs=brain_list_state, 
        outputs=[brain_cards, brain_selector]
    )
    
    # Select brain from cards (for deletion)
    # Dataset.select automatically passes the selected row as a SelectData object
    def handle_brain_selection(evt: gr.SelectData, brains_list):
        """Extract brain name from selected row"""
        if brains_list and evt.index < len(brains_list):
            return brains_list[evt.index][0]
        return ""
    
    brain_cards.select(
        fn=handle_brain_selection,
        inputs=[brain_list_state],
        outputs=selected_brain_to_delete
    )
    
    # Delete brain and update all components
    def delete_brain_and_update(brain_name):
        status, updated_list = delete_brain(brain_name)
        brain_names = [b[0] for b in updated_list]
        return (
            status,                                  # manage_status
            updated_list,                            # brain_list_state
            "",                                      # clear selected_brain_to_delete
            gr.update(samples=updated_list),         # update brain_cards
            gr.update(choices=brain_names)           # update brain_selector dropdown
        )
    
    delete_brain_button.click(
        fn=delete_brain_and_update,
        inputs=selected_brain_to_delete,
        outputs=[manage_status, brain_list_state, selected_brain_to_delete, brain_cards, brain_selector]
    )
    
    # Task 2: Update file list and clear chat when brain is selected in chat page
    def switch_brain(brain_name):
        files = get_files_in_brain(brain_name)
        return files, [], ""  # Clear chatbot, clear status
    
    brain_selector.change(
        fn=switch_brain,
        inputs=brain_selector,
        outputs=[file_list, chatbot, ingest_status]
    )
    
    # Upload files to brain
    ingest_button.click(
        fn=ingest_files,
        inputs=[file_input, brain_selector],
        outputs=[ingest_status, file_list]
    )
    
    # Voice input: transcribe audio and populate text box
    audio_input.change(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=msg_box
    )
    
    # Task 3: Chat with brain and clear input box
    def chat_and_clear(user_message, chat_history, brain_name):
        # Generate response
        for updated_history in chat_with_brain(user_message, chat_history, brain_name):
            yield updated_history, ""  # Clear input box while streaming
    
    msg_box.submit(
        fn=chat_and_clear,
        inputs=[msg_box, chatbot, brain_selector],
        outputs=[chatbot, msg_box]
    )

# --- 6. LAUNCH THE APP ---
if __name__ == "__main__":
    print("--- Launching Gradio UI ---")
    demo.launch(share=False, server_port=7860)

