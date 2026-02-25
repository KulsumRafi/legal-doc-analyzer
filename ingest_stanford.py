"""
Stanford Material Contracts Corpus Ingestion
Processes downloaded MCC files and stores in ChromaDB
"""

import os
import glob
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from bs4 import BeautifulSoup

# Configuration
MCC_FOLDER = "./data/mcc_download"
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "stanford_mcc"

def extract_text_from_html(file_path):
    """Extract clean text from HTML contract files"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def load_stanford_contracts():
    """Load contracts from Stanford MCC folder"""
    documents = []
    
    # Check if folder exists
    if not os.path.exists(MCC_FOLDER):
        print(f"📁 Creating folder: {MCC_FOLDER}")
        os.makedirs(MCC_FOLDER)
        print(f"⚠️ Please download Stanford MCC files and place them in {MCC_FOLDER}")
        return []
    
    # Find all .htm files
    file_patterns = ["*.htm", "*.html"]
    file_count = 0
    
    for pattern in file_patterns:
        for file_path in glob.glob(os.path.join(MCC_FOLDER, pattern)):
            file_count += 1
            print(f"Loading [{file_count}]: {os.path.basename(file_path)}")
            
            try:
                # Extract text from HTML
                text = extract_text_from_html(file_path)
                
                # Try to determine contract type from filename
                filename = os.path.basename(file_path).lower()
                contract_type = "Unknown"
                if "employ" in filename:
                    contract_type = "Employment"
                elif "merger" in filename or "acquisition" in filename:
                    contract_type = "M&A"
                elif "lease" in filename:
                    contract_type = "Lease"
                elif "credit" in filename or "loan" in filename:
                    contract_type = "Security"
                elif "service" in filename:
                    contract_type = "Services"
                
                documents.append(Document(
                    page_content=text[:50000],  # Limit size
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": contract_type,
                        "source_type": "stanford_mcc",
                        "file_path": file_path
                    }
                ))
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Loaded {len(documents)} contracts from Stanford MCC")
    return documents

def ingest_stanford():
    """Main Stanford ingestion function"""
    print("="*60)
    print("🔵 Stanford Material Contracts Corpus Ingestion")
    print("="*60)
    
    # Step 1: Load contracts
    print("\n📂 Loading Stanford MCC contracts...")
    contracts = load_stanford_contracts()
    
    if not contracts:
        print("\n❌ No contracts found. Please:")
        print(f"   1. Download files from https://mcc.law.stanford.edu")
        print(f"   2. Place them in: {MCC_FOLDER}")
        print(f"   3. Run this script again")
        return
    
    # Step 2: Split into chunks
    print("\n✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(contracts)
    print(f"   Created {len(chunks)} chunks from {len(contracts)} contracts")
    
    # Step 3: Create embeddings
    print("\n🔤 Creating embeddings (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Step 4: Store in ChromaDB (Chroma CREATES the folder automatically)
    print(f"\n💾 Storing in ChromaDB: {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Step 5: Persist
    vectordb.persist()
    
    # Summary
    print("\n" + "="*60)
    print("✅ STANFORD INGESTION COMPLETE")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Contracts processed: {len(contracts)}")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Database: {CHROMA_DB_PATH}")
    print(f"   - Collection: {COLLECTION_NAME}")
    print("="*60)

if __name__ == "__main__":
    ingest_stanford()