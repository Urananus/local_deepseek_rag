import asyncio
from pathlib import Path
import os 
from typing import List
from pydantic import BaseModel, Field, ValidationError, model_validator
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import ollama

class RAGConfig(BaseModel):
    """Optimized CPU configuration with model validation"""
    data_dir: Path = Field(..., description="Path to document directory")
    chunk_size: int = Field(512, ge=256, le=1024, description="Optimal CPU chunk size")
    chunk_overlap: int = Field(128, ge=0, description="Context overlap")
    embeddings_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model name",
        env="RAG_MODEL"  # Add environment variable binding
    )
    search_k: int = Field(3, description="Retrieval count")
    ollama_timeout: int = Field(60, description="Embedding request timeout")

    @model_validator(mode='after')
    def validate_ollama(self):
        """Verify Ollama service and model availability"""
        try:
            # Check connection by listing models
            model_list = ollama.list()
            models = model_list.get('models', [])
            
            if not models:
                raise ValueError("No Ollama models installed. Install first with:\nollama pull nomic-embed-text")

            # Check for model name with version tag handling
            model_exists = any(
                self.embeddings_model in m.model
                for m in models
            )
            
            if not model_exists:
                available_models = "\n".join([f"- {m.model}" for m in models])
                raise ValueError(
                    f"Model '{self.embeddings_model}' not found.\n"
                    f"Installed models:\n{available_models}\n"
                    f"Install with: ollama pull {self.embeddings_model}"
                )
                
        except Exception as e:
            raise ValueError(
                f"Ollama connection failed: {str(e)}\n"
                "First troubleshooting steps:\n"
                "1. Start Ollama service: ollama serve\n"
                "2. Verify installation: ollama list"
            ) from e
        
        return self

async def load_documents(config: RAGConfig) -> List[Document]:
    """Async document loading with parallel processing"""
    loader = DirectoryLoader(
        str(config.data_dir),
        show_progress=True,
        use_multithreading=True,
        silent_errors=True
    )
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)
    
    return await splitter.atransform_documents(await loader.aload())

async def create_retriever(config: RAGConfig) -> FAISS.as_retriever:
    """Create optimized retriever with error handling"""
    try:
        embeddings = OllamaEmbeddings(
            model=config.embeddings_model,
            base_url="http://localhost:11434"
        )
        
        documents = await load_documents(config)
        return FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        ).as_retriever(search_kwargs={"k": config.search_k})
    
    except Exception as e:
        raise RuntimeError(f"Retriever creation failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Remove hardcoded model name to use environment variable
        config = RAGConfig(
            data_dir=Path("/home/jesus/local_deepseek_rag/myDocuments"),
            chunk_size=512,
            chunk_overlap=128,
            search_k=3,
            ollama_timeout=60
        )
        retriever = asyncio.run(create_retriever(config))
        
        print("🦙 CPU-Optimized RAG System Ready")
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    break
                
                results = retriever.get_relevant_documents(query)
                print(f"\n📚 Retrieved {len(results)} results:")
                for i, doc in enumerate(results, 1):
                    source = Path(doc.metadata['source']).name
                    print(f"{i}. {source} ({len(doc.page_content)} chars)")
                    print(f"   {doc.page_content[:296]}...\n")
                    print("🦙")
                    
            except KeyboardInterrupt:
                print("\n🛑 Operation cancelled")
                break
            
    except ValidationError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"🔥 Critical error: {str(e)}")