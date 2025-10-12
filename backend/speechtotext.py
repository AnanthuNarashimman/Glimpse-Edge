import vosk
import pyaudio
import json
import os
import sys

# --- Configuration ---
# IMPORTANT: Update this to the path where you saved your Vosk model folder
VOSK_MODEL_PATH = "vosk-model-en-us-0.22-lgraph" 
SAMPLE_RATE = 16000
CHUNK_SIZE = 8192

# --- Voice Mode Triggers ---
# Use a short, distinct phrase to reliably turn the module ON
WAKE_PHRASE = "hello" 
TERMINATE_PHRASE = "terminate program" # To shut down the script entirely

# --- Core Functions ---

def text_to_query(text: str) -> dict:
    """
    Converts transcribed text into a structured command/query dictionary.
    """
    text_lower = text.lower().strip()

    # Self-commands (checked first)
    if TERMINATE_PHRASE in text_lower:
        return {"command": "terminate", "text": text_lower}
    elif text_lower == WAKE_PHRASE:
         return {"command": "wake", "text": text_lower} # Check for exact wake word match
    
    # User Commands
    if text_lower.startswith("search for"):
        term = text_lower.replace("search for", "", 1).strip()
        return {"command": "search", "term": term}
    
    elif "set a timer for" in text_lower:
        parts = text_lower.split("set a timer for")
        duration = parts[1].strip() if len(parts) > 1 else "unspecified"
        return {"command": "timer", "duration": duration}

    else:
        # Default for all other commands
        return {"command": "unrecognized", "text": text_lower}


def offline_voice_to_query():
    """
    Main function to handle microphone input, implementing the ON-Command-OFF cycle.
    """
    # Initialize state variable
    active_listening = False
    
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Error: Vosk model not found at {VOSK_MODEL_PATH}")
        print("Please download a model and update the VOSK_MODEL_PATH variable.")
        sys.exit(1)

    try:
        # Initialize Vosk model and recognizer
        model = vosk.Model(VOSK_MODEL_PATH)
        rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        
        # Setup PyAudio for microphone input
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=SAMPLE_RATE, 
            input=True, 
            frames_per_buffer=CHUNK_SIZE
        )
        stream.start_stream()

        print("\n--- Voice Module Initialized ---")
        print(f"Status: 🔴 Sleeping. Say '{WAKE_PHRASE}' to wake the module.")
        
        # Setup for console editing
        PROMPT = "🎙️ Listening: "
        
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # --- 1. Get Final Result (User paused/completed speech) ---
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recognized_text = result.get('text', '').strip()
                
                # --- Wake Word Check (Even if sleeping, check for wake word) ---
                if not active_listening:
                    if recognized_text.lower() == WAKE_PHRASE:
                        active_listening = True
                        print("\r" + " " * 80 + "\r", end="") # Clear the line
                        print("\n\n--- WAKE WORD DETECTED ---")
                        print("Status: 🟢 Active. Ready for command.")
                        sys.stdout.write(PROMPT)
                        sys.stdout.flush()
                    continue 

                # --- Active Mode: Process Command and Auto-Off ---
                if active_listening and recognized_text:
                    # Clear the partial transcription line
                    sys.stdout.write('\r' + ' ' * 80 + '\r') 
                    
                    query = text_to_query(recognized_text)
                    
                    print("-" * 30)
                    print(f"✅ Transcription Complete (Editable Text): {recognized_text}")
                    print(f"💡 Structured Query: {query}")
                    print("-" * 30)
                    
                    # 💡 Requirement: AUTO-OFF after one command
                    active_listening = False 
                    print(f"Status: 🔴 Sleeping (Auto-Off). Say '{WAKE_PHRASE}' to resume.")

                    if query.get("command") == "terminate":
                        break
            
            # --- 2. Partial Result (User is speaking while active) ---
            else:
                partial_result = json.loads(rec.PartialResult()).get('partial', '')
                
                if active_listening:
                    # Write the partial text, overwriting the previous one
                    sys.stdout.write('\r' + ' ' * 80 + '\r') 
                    sys.stdout.write(f"{PROMPT}{partial_result}")
                    sys.stdout.flush()
                else:
                    # In sleep mode, just process the stream silently
                    continue


    except KeyboardInterrupt:
        print("\nExiting module.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup PyAudio and stream
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()


# --- Execution ---
if __name__ == "__main__":
    offline_voice_to_query()