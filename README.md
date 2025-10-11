# RAG-Powered Research Knowledge Assistant

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.0-orange.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.22-purple.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA-red.svg)](https://groq.com/)

> **Intelligent Q&A System for Machine Learning Research Documents**

A production-ready LangChain-powered chatbot that answers questions about your research documents using semantic search with ChromaDB vector database and Groq's LLaMA model. Built for researchers, students, and ML practitioners who need instant access to knowledge from their document collections.

## Features

- 📄 Load and process multiple research documents (.txt files)
- 🔍 Semantic search using sentence transformers
- 💬 Interactive chat interface
- 🎯 Context-aware responses based on your documents
- ⚡ Fast retrieval using ChromaDB vector database
- 🔒 Production-ready with comprehensive error handling

## 📚 Document Domain

### Supported Document Types

This system is specifically designed and optimized for:

- **Primary Focus**: Academic papers, research publications, technical documentation
- **Domain**: Machine Learning, Artificial Intelligence, Data Science, Computer Science
- **Format**: Text files (.txt) with extensibility to PDF, DOCX
- **Content Characteristics**: Long-form technical content with structured information

### Best Suited For

✅ **Technical and academic research papers** - ML/AI research, conference papers, journal articles
✅ **Machine learning literature** - Papers on algorithms, architectures, methodologies
✅ **Technical documentation** - API docs, implementation guides, technical whitepapers
✅ **Educational materials** - Course notes, textbooks, tutorial content
✅ **Knowledge base** - Company documentation, research team knowledge repositories

### Not Recommended For

❌ Creative writing or fiction
❌ News articles or blog posts
❌ Social media content
❌ Unstructured conversational text
❌ Real-time or time-sensitive information

### Document Characteristics

- **Length**: Works best with documents > 500 words
- **Structure**: Benefits from well-organized content with clear sections
- **Language**: Optimized for English technical content
- **Format**: Currently supports .txt files (see roadmap for PDF/DOCX support)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy and edit the `.env` file with your API keys:
```bash
# Required: Get your Groq API key from https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Add Research Documents
Place your research papers (.txt files) in the `documents/` folder:
```bash
documents/
├── paper1.txt
├── paper2.txt
└── paper3.txt
```

### 4. Run the Chatbot

**Option 1: Clean startup (recommended)**
```bash
python start_chatbot.py
```

**Option 2: Direct run**
```bash
python rt_lc_chatbot.py
```

> Note: The clean startup script suppresses ChromaDB initialization messages for a better user experience.

## Usage

1. **First run**: The system will automatically process and index your documents
2. **Ask questions**: Type questions about your research documents
3. **Get answers**: The AI will provide answers with source citations
4. **Exit**: Type 'quit', 'exit', or 'q' to stop

### Example Questions
- "What are effective techniques for handling class imbalance?"
- "How does SMOTE work?"
- "What evaluation metrics should I use for imbalanced datasets?"
- "Compare different oversampling techniques"

## Configuration

Edit `.env` file to customize:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)  
- `TOP_K_RESULTS`: Number of relevant chunks to retrieve (default: 3)
- `LLM_MODEL_NAME`: Groq model to use (default: llama-3.1-8b-instant)

## Troubleshooting

### Common Issues

1. **Missing Groq API Key**
   - Get your API key from https://console.groq.com/
   - Add it to the `.env` file

2. **No Documents Found**
   - Make sure `.txt` files are in the `documents/` folder
   - Check file permissions

3. **Model Deprecation Error**
   - Update the `LLM_MODEL_NAME` in `.env` to a supported model
   - Check https://console.groq.com/docs/models for current models

4. **Tokenizer Parallelism Warning**
   - Already fixed with `TOKENIZERS_PARALLELISM=false` in `.env`

## File Structure
```
.
├── rt_lc_chatbot.py      # Main chatbot script
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── documents/            # Your research documents (.txt)
├── research_db/          # ChromaDB vector database (auto-created)
└── README.md            # This file
```

## Dependencies

- **ChromaDB**: Vector database for document storage and retrieval
- **LangChain**: Framework for building LLM applications
- **Groq**: Fast LLM inference
- **Sentence Transformers**: Text embedding models
- **PyTorch**: Deep learning framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/) for LLM orchestration
- Powered by [ChromaDB](https://www.trychroma.com/) for vector storage
- Uses [Groq](https://groq.com/) for fast LLM inference
- Embeddings from [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)

## 📬 Contact & Support

For issues, questions, or contributions, please visit the project repository or open an issue.

---

**Built for the AI research community** 🚀
