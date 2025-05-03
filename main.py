from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import os
import requests
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Dict, Any
from pathlib import Path

class OllamaLocalLLM(LLM):
    model: str = "llama3:latest"  # Updated to match the exact model name with tag
    base_url: str = "http://localhost:11434"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "stop": stop
                },
                timeout=60
            )
            
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama API: {e}")
    
    @property
    def _llm_type(self) -> str:
        return "ollama-local"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}

def load_book(file_path: str):
    """Load book content from PDF or TXT"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt allowed")
    
    return loader.load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split the documents"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_chroma_collection(book_name, chunks):
    """Create ChromaDB collection and store chunks"""
    persist_directory = f"./chroma_store/{book_name}"
    
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Try different parameter names for compatibility
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,  # Try with 'embedding' first
            persist_directory=persist_directory,
            collection_name=book_name
        )
    except TypeError:
        # If that fails, try with embedding_function
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name=book_name
        )
    
    vectordb.persist()
    return vectordb

def create_qa_chain(book_name, model_name):
    """Create QA chain with retriever and custom Ollama HTTP wrapper"""
    persist_directory = f"./chroma_store/{book_name}"
    
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Try different parameter names for compatibility
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding=embedding,  # Try with 'embedding' first
            collection_name=book_name
        )
    except TypeError:
        # If that fails, try with embedding_function
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
            collection_name=book_name
        )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use the model_name parameter passed in
    llm = OllamaLocalLLM(model=model_name)
    
    # Create a basic QA chain using the older API
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    
    # This is a simple function to combine retrieval and QA
    def run_qa(query):
        try:
            # Try the new invoke method first (newer LangChain)
            docs = retriever.invoke(query)
            return qa_chain.invoke({"input_documents": docs, "question": query})
        except AttributeError:
            # Fall back to the old method
            docs = retriever.get_relevant_documents(query)
            return qa_chain({"input_documents": docs, "question": query})
    
    return run_qa

def check_ollama_connection():
    """Check if Ollama is running at the specified port and get available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        available_models = [model.get("name") for model in models]
        return True, available_models
    except requests.exceptions.RequestException:
        return False, []

def get_base_model_name(model_with_tag):
    """Extract base model name without tag"""
    return model_with_tag.split(':')[0]

def main():
    # Check if required packages are installed
    try:
        import langchain_community
        import langchain_text_splitters
    except ImportError:
        print("❌ Missing required packages. Please install them with:")
        print("pip install langchain langchain-community langchain-text-splitters")
        return
    
    # Check Ollama connection first
    print("🔍 Checking Ollama connection...")
    ollama_running, available_models = check_ollama_connection()
    
    if not ollama_running:
        print("\n❌ Ollama is not running at http://localhost:11434.")
        print("Please make sure to:")
        print("1. Install Ollama from https://ollama.ai/")
        print("2. Run the Ollama server with 'ollama serve'")
        print("3. Pull a language model with 'ollama pull llama3'")
        return
    
    print("✅ Ollama is running!")
    if available_models:
        print(f"📚 Available models: {', '.join(available_models)}")
        
        # Choose a model to use
        default_model = available_models[0]  # Use the first available model as default
        print(f"\nUsing model: {default_model}")
    else:
        print("❌ No models found. Please pull a model first:")
        print("ollama pull llama3")
        return
        
    print("\n📘 Upload your book (.pdf or .txt):")
    file_path = input("Enter file path: ").strip()

    if not os.path.isfile(file_path):
        print("❌ File does not exist.")
        return

    book_name = Path(file_path).stem

    print("📖 Loading book...")
    try:
        docs = load_book(file_path)
    except Exception as e:
        print(f"❌ Error loading book: {e}")
        return

    print("✂️ Splitting book into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print(f"📚 Creating ChromaDB collection for '{book_name}'...")
    try:
        create_chroma_collection(book_name, chunks)
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        return

    print("✅ Book uploaded and indexed successfully!")

    try:
        # Use the selected model
        qa_function = create_qa_chain(book_name, default_model)
    except Exception as e:
        print(f"❌ Error creating QA chain: {e}")
        return

    print("\n🔍 Ready to answer questions about your book!")
    while True:
        query = input("\n❓ Ask a question about the book (or type 'exit' to quit):\n> ")
        if query.lower() == 'exit':
            break
        
        try:
            response = qa_function(query)
            # Handle different output key formats
            output_key = 'answer' if 'answer' in response else 'output_text'
            print("\n🤖 Answer:")
            print(response[output_key])
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()