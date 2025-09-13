#!/usr/bin/env python3
"""
Clean startup script for the Research Assistant Chatbot
Suppresses all ChromaDB warnings and telemetry errors
"""

import os
import sys
import warnings

# Set environment variables before importing anything
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_CLIENT_AUTH_PROVIDER"] = ""
os.environ["CHROMA_SERVER_AUTH_PROVIDER"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress logging
import logging
logging.basicConfig(level=logging.ERROR)
for logger_name in ["chromadb", "chromadb.telemetry", "chromadb.api", "sentence_transformers"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Redirect stderr to suppress chromadb messages
class NullWriter:
    def write(self, txt): pass
    def flush(self): pass

original_stderr = sys.stderr

def suppress_output():
    sys.stderr = NullWriter()

def restore_output():
    sys.stderr = original_stderr

# Import and run the main chatbot
if __name__ == "__main__":
    print("🚀 Starting Research Assistant Chatbot...")
    print("   (Initializing ChromaDB - this may take a moment...)")
    
    # Temporarily suppress output during ChromaDB initialization
    suppress_output()
    
    try:
        from rt_lc_chatbot import main
        # Restore output before running main
        restore_output()
        main()
    except ImportError as e:
        restore_output()
        print(f"❌ Error importing chatbot: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except KeyboardInterrupt:
        restore_output()
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        restore_output()
        print(f"❌ Error running chatbot: {e}")