"""
RAG Research Assistant - Query research publications using LLM
"""

import chromadb
import os
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "research_db")

# Initialize ChromaDB
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)

# GPU setup
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device} if device == "cuda" else {}
)

def load_research_publications(documents_path):
    """Load research publications from directory (supports .txt and .pdf)"""
    publications = []
    
    if not os.path.exists(documents_path):
        print(f"ERROR: Directory does not exist: {documents_path}")
        return publications
    
    all_files = os.listdir(documents_path)
    txt_files = [f for f in all_files if f.lower().endswith(".txt")]
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
    
    # Load text files
    for file in txt_files:
        file_path = os.path.join(documents_path, file)
        try:
            loader = TextLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                publications.append({
                    "content": doc.page_content,
                    "title": os.path.splitext(file)[0]
                })
            print(f"✓ Loaded: {file}")
        except Exception as e:
            print(f"✗ Error loading {file}: {str(e)}")
    
    # Load PDF files
    for file in pdf_files:
        file_path = os.path.join(documents_path, file)
        try:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            full_content = "\n\n".join([doc.page_content for doc in loaded_docs])
            publications.append({
                "content": full_content,
                "title": os.path.splitext(file)[0]
            })
            print(f"✓ Loaded: {file} ({len(loaded_docs)} pages)")
        except Exception as e:
            print(f"✗ Error loading {file}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(publications)}")
    return publications

def chunk_research_paper(paper_content, title):
    """Chunk a research paper into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(paper_content)
    return [
        {
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}"
        }
        for i, chunk in enumerate(chunks)
    ]

def embed_documents(documents: list[str]) -> list[list[float]]:
    """Embed documents using HuggingFaceEmbeddings"""
    return embeddings.embed_documents(documents)

def insert_publications(collection: chromadb.Collection, publications: list[dict]):
    """Insert publications into the collection"""
    next_id = collection.count()
    
    for publication in publications:
        chunked_data = chunk_research_paper(publication["content"], publication["title"])
        chunk_contents = [chunk["content"] for chunk in chunked_data]
        chunk_embeddings = embed_documents(chunk_contents)
        
        num_chunks = len(chunked_data)
        ids = [f"document_{next_id + i}" for i in range(num_chunks)]
        metadatas = [
            {"title": chunk["title"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunked_data
        ]
        
        collection.add(
            embeddings=chunk_embeddings,
            ids=ids,
            documents=chunk_contents,
            metadatas=metadatas
        )
        next_id += num_chunks
    
    print(f"✓ Inserted {len(publications)} publications into collection")

def search_research_db(query, collection, embeddings, top_k=5):
    """Search the research database for relevant publications"""
    query_vector = embeddings.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    return [
        {
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]

def answer_research_question(query, collection, embeddings, llm):
    """Answer a research question using the research database"""
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Based on the following research findings, answer the researcher's question:

        Research Context:
        {context}

        Researcher's Question: {question}

        Answer: Provide a comprehensive answer based on the research findings above."""
    )
    
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    
    return response.content, relevant_chunks

if __name__ == "__main__":
    documents_path = os.path.join(script_dir, "data")
    
    print(f"\nDocuments path: {documents_path}")
    print(f"Collection contains: {collection.count()} documents\n")
    
    # Load publications if collection is empty
    if collection.count() == 0:
        if os.path.exists(documents_path):
            publications = load_research_publications(documents_path)
            if publications:
                insert_publications(collection, publications)
            else:
                print(f"No documents found in {documents_path}")
                exit(0)
        else:
            print(f"Documents path does not exist: {documents_path}")
            exit(0)
    
    # Initialize Groq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=groq_model,
            temperature=0.7
        )
        print(f"✓ LLM initialized: {groq_model}\n")
    except Exception as e:
        print(f"✗ Error initializing LLM: {str(e)}")
        exit(1)
    
    # Query
    query = "What are the topics of machine behavior in the field of Information Systems?"
    print(f"Query: {query}\n")
    
    answer, sources = answer_research_question(query, collection, embeddings, llm)
    
    print("="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    print("\n" + "="*80)
    print("SOURCES:")
    print("="*80)
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] {source['title']} (similarity: {source['similarity']:.4f})")
        print(f"    {source['content'][:200]}...")
