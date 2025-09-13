import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import os
from langchain_community.document_loaders import TextLoader

import torch
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry and warnings completely
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_CLIENT_AUTH_PROVIDER"] = ""
os.environ["CHROMA_SERVER_AUTH_PROVIDER"] = ""

# Suppress all ChromaDB warnings and errors
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Initialize ChromaDB with all telemetry disabled
try:
    # Create settings with telemetry completely disabled
    settings = chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
    client = chromadb.PersistentClient(
        path=os.getenv('CHROMA_DB_PATH', './research_db'),
        settings=settings
    )
except Exception:
    # Fallback to basic client with minimal settings
    try:
        client = chromadb.PersistentClient(
            path=os.getenv('CHROMA_DB_PATH', './research_db')
        )
    except Exception as e:
        print(f"⚠️ ChromaDB initialization warning (can be ignored): {e}")
        # Create client anyway
        client = chromadb.PersistentClient(path=os.getenv('CHROMA_DB_PATH', './research_db'))
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

# Set up our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
)



def load_research_publications(documents_path):
    """Load research publications from .txt files and return as list of strings"""
    
    # List to store all documents
    documents = []
    
    # Load each .txt file in the documents folder
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Extract content as strings and return
    publications = []
    for doc in documents:
        publications.append(doc.page_content)
    
    return publications



def chunk_research_paper(paper_content, title):
    """Break a research paper into searchable chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(paper_content)
    
    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    
    return chunk_data



def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name=os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def chunk_publication(publication_content):
    """Break a publication into searchable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200)),
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(publication_content)

def insert_publications(collection: chromadb.Collection, publications: list[str]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        publications (list[str]): The documents to insert

    Returns:
        None
    """
    for i, publication in enumerate(publications):
        chunked_publication = chunk_publication(publication)
        embeddings_list = embed_documents(chunked_publication)
        
        # Create metadata for each chunk
        metadatas = []
        ids = []
        for j in range(len(chunked_publication)):
            chunk_id = f"doc_{i}_chunk_{j}"
            ids.append(chunk_id)
            metadatas.append({
                "title": f"Document_{i+1}",
                "chunk_id": chunk_id,
                "source": f"document_{i+1}.txt"
            })
        
        collection.add(
            embeddings=embeddings_list,
            ids=ids,
            documents=chunked_publication,
            metadatas=metadatas
        )
        print(f"✅ Processed document {i+1}/{len(publications)} with {len(chunked_publication)} chunks")



def search_research_db(query, collection, embeddings, top_k=5):
    """Find the most relevant research chunks for a query"""
    
    # Check if collection has any documents
    if collection.count() == 0:
        print("⚠️ No documents in the database. Please add documents first.")
        return []
    
    # Convert question to vector
    query_vector = embeddings.embed_query(query)
    
    # Adjust top_k to not exceed available documents
    actual_count = collection.count()
    if top_k > actual_count:
        top_k = actual_count
        print(f"ℹ️ Adjusted search results to {top_k} (total available documents)")
    
    # Search for similar content
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Check if we got results
        if not results["documents"] or not results["documents"][0]:
            return []
        
        # Format results
        relevant_chunks = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
            title = metadata.get("title", f"Document_{i+1}")
            
            relevant_chunks.append({
                "content": doc,
                "title": title,
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return relevant_chunks
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []



def answer_research_question(query, collection, embeddings, llm):
    """Generate an answer based on retrieved research"""
    
    print(f"🔍 Processing question: '{query}'")
    
    # Get relevant research chunks
    top_k = int(os.getenv('TOP_K_RESULTS', 3))
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=top_k)
    
    if not relevant_chunks:
        return "❌ No relevant information found in the database.", []
    
    # Build context from research
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}" 
        for chunk in relevant_chunks
    ])
    
    # Create research-focused prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a research assistant AI. Based on the research context provided below, answer the user's question accurately and comprehensively.

Research Context:
{context}

User's Question: {question}

Instructions:
- Provide a detailed answer based only on the research context above
- If the context doesn't contain enough information, say so clearly
- Include specific details and examples from the research when relevant
- Be concise but thorough

Answer:
"""
    )
    
    # Generate answer
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    
    return response.content, relevant_chunks

def load_and_process_documents(documents_path="./documents"):
    """Load and process documents into the ChromaDB collection"""
    if not os.path.exists(documents_path):
        print(f"⚠️ Documents folder '{documents_path}' not found.")
        print("📁 Please create a 'documents' folder and add .txt files to it.")
        return False
    
    # Check for txt files
    txt_files = [f for f in os.listdir(documents_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"⚠️ No .txt files found in '{documents_path}'.")
        print("📄 Please add some .txt research documents to the folder.")
        return False
    
    # Load publications
    publications = load_research_publications(documents_path)
    
    if not publications:
        print("❌ No documents could be loaded.")
        return False
    
    # Insert into database if collection is empty
    if collection.count() == 0:
        print(f"📚 Processing and storing {len(publications)} documents...")
        insert_publications(collection, publications)
        print(f"✅ Successfully processed {len(publications)} documents into {collection.count()} chunks.")
    else:
        print(f"📊 Database already contains {collection.count()} document chunks.")
    
    return True

def main():
    """Main function to run the research chatbot"""
    # Initialize LLM
    llm = ChatGroq(
        model=os.getenv('LLM_MODEL_NAME', 'llama-3.1-8b-instant'),
        groq_api_key=os.getenv('GROQ_API_KEY')
    )
    
    # Load documents if needed
    load_and_process_documents()
    
    # Interactive chatbot loop
    print("\n=== Research Assistant Chatbot ===")
    print("Ask questions about your research documents. Type 'quit' to exit.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            answer, sources = answer_research_question(
                query,
                collection, 
                embeddings, 
                llm
            )
            
            print(f"\n🤖 Answer: {answer}")
            print("\n📚 Sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source['title']} (similarity: {source['similarity']:.2f})")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("-" * 80)

if __name__ == "__main__":
    main()
