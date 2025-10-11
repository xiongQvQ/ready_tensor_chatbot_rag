#!/usr/bin/env python3
"""
Quick test script to verify the RAG system is working
"""

import sys
import os

# Suppress ChromaDB warnings
import warnings
warnings.filterwarnings('ignore')

try:
    print("🧪 Testing RAG System...\n")

    # Test 1: Import all dependencies
    print("✓ Test 1: Importing dependencies...")
    import chromadb
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("  ✅ All dependencies imported successfully\n")

    # Test 2: Load environment variables
    print("✓ Test 2: Loading environment variables...")
    from dotenv import load_dotenv
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("  ❌ GROQ_API_KEY not found in .env")
        sys.exit(1)
    print("  ✅ Environment variables loaded\n")

    # Test 3: Initialize embeddings
    print("✓ Test 3: Initializing embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )
    test_embedding = embedding_model.embed_query("test")
    print(f"  ✅ Embeddings initialized (dim: {len(test_embedding)})\n")

    # Test 4: Check documents folder
    print("✓ Test 4: Checking documents folder...")
    docs_path = "documents"
    if not os.path.exists(docs_path):
        print(f"  ❌ Documents folder not found: {docs_path}")
        sys.exit(1)

    doc_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
    if not doc_files:
        print(f"  ⚠️  No .txt files found in {docs_path}")
    else:
        print(f"  ✅ Found {len(doc_files)} document(s): {', '.join(doc_files)}\n")

    # Test 5: Initialize ChromaDB
    print("✓ Test 5: Initializing ChromaDB...")
    chroma_client = chromadb.Client()
    print("  ✅ ChromaDB initialized\n")

    # Test 6: Initialize LLM
    print("✓ Test 6: Initializing LLM...")
    llm = ChatGroq(
        model_name=os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant"),
        temperature=0,
        groq_api_key=groq_api_key
    )
    print("  ✅ LLM initialized\n")

    print("=" * 60)
    print("🎉 SUCCESS! All system components are working correctly!")
    print("=" * 60)
    print("\n✅ System is ready to use")
    print("✅ Run 'python start_chatbot.py' to start the chatbot")

except Exception as e:
    print(f"\n❌ Test failed with error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
