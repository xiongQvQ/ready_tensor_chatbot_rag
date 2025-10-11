#!/usr/bin/env python3
"""
Integration test: Run a real query through the complete RAG system
"""

import sys
import os
from io import StringIO

# Redirect stderr to suppress ChromaDB warnings during initialization
stderr_backup = sys.stderr
sys.stderr = StringIO()

try:
    from dotenv import load_dotenv
    from rt_lc_chatbot import (
        collection,
        embeddings,
        load_and_process_documents,
        answer_research_question
    )
    from langchain_groq import ChatGroq

    # Restore stderr after imports
    sys.stderr = stderr_backup

    print("🧪 Integration Test: Complete RAG Pipeline\n")
    print("=" * 60)

    # Load environment
    load_dotenv()

    # Step 1: Load and process documents
    print("\n📚 Step 1: Loading and processing documents...")
    success = load_and_process_documents("documents")
    if not success:
        print("❌ Failed to load documents")
        sys.exit(1)
    print(f"   ✅ Documents processed")

    # Step 2: Initialize LLM
    print("\n🤖 Step 2: Initializing LLM...")
    llm = ChatGroq(
        model=os.getenv('LLM_MODEL_NAME', 'llama-3.1-8b-instant'),
        groq_api_key=os.getenv('GROQ_API_KEY')
    )
    print("   ✅ LLM initialized")

    # Step 3: Test query
    print("\n💬 Step 3: Testing query...")
    test_query = "What is artificial intelligence?"
    print(f"   Query: '{test_query}'")

    # Generate answer
    print("\n🔍 Step 4: Generating answer with RAG system...")
    answer, sources = answer_research_question(test_query, collection, embeddings, llm)

    # Display results
    print("\n" + "=" * 60)
    print("📋 QUERY RESULT:")
    print("=" * 60)
    print(f"\nQuestion: {test_query}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n📚 Sources ({len(sources)}):")
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source['title']} (similarity: {source['similarity']:.2f})")

    print("\n" + "=" * 60)
    print("\n✅ Integration test PASSED!")
    print("🎉 Complete RAG system is fully functional!")
    print("\n💡 Run 'python start_chatbot.py' to start interactive mode")

except Exception as e:
    # Restore stderr in case of error
    sys.stderr = stderr_backup
    print(f"\n❌ Integration test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
