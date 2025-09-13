# Research Assistant Chatbot

A LangChain-powered chatbot that can answer questions about your research documents using ChromaDB vector database and Groq's LLaMA model.

## Features

- 📄 Load and process multiple research documents (.txt files)
- 🔍 Semantic search using sentence transformers
- 💬 Interactive chat interface
- 🎯 Context-aware responses based on your documents
- ⚡ Fast retrieval using ChromaDB vector database

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

## License

MIT License