"""
Stanford Material Contracts Corpus Ingestion
Optimized for ~3GB dataset
"""

import os
import glob
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from bs4 import BeautifulSoup
import time
from tqdm import tqdm  # For progress bars (pip install tqdm)

# Configuration
MCC_FOLDER = "./data/mcc_download"
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "stanford_mcc"

# Create output folder
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

def extract_text_from_html(file_path):
    """Extract clean text from HTML contract files"""
    try:
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
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def load_stanford_contracts():
    """Load contracts from Stanford MCC folder"""
    documents = []
    
    # Find all files
    file_patterns = ["*.htm", "*.html", "*.txt"]
    all_files = []
    
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(MCC_FOLDER, pattern)))
    
    print(f"📂 Found {len(all_files)} files to process")
    
    if not all_files:
        print(f"❌ No files found in {MCC_FOLDER}")
        print("Please place your Stanford MCC files there.")
        return []
    
    # Process with progress bar
    for file_path in tqdm(all_files, desc="Processing contracts"):
        filename = os.path.basename(file_path)
        
        # Extract text
        text = extract_text_from_html(file_path)
        
        if len(text) < 100:  # Skip empty files
            continue
        
        # Try to determine contract type from filename
        filename_lower = filename.lower()
        contract_type = "Other"
        
        if any(word in filename_lower for word in ["employ", "compensation", "severance"]):
            contract_type = "Employment"
        elif any(word in filename_lower for word in ["merger", "acquisition", "merger agreement"]):
            contract_type = "M&A"
        elif any(word in filename_lower for word in ["lease", "rental"]):
            contract_type = "Lease"
        elif any(word in filename_lower for word in ["credit", "loan", "security", "note"]):
            contract_type = "Security"
        elif any(word in filename_lower for word in ["service", "consulting", "professional"]):
            contract_type = "Services"
        
        documents.append(Document(
            page_content=text[:50000],  # Limit size for consistency
            metadata={
                "source": filename,
                "type": contract_type,
                "source_type": "stanford_mcc",
                "file_path": file_path,
                "size_chars": len(text)
            }
        ))
    
    print(f"\n✅ Loaded {len(documents)} contracts")
    return documents

def ingest_stanford():
    """Main Stanford ingestion function"""
    print("="*60)
    print("🔵 Stanford Material Contracts Corpus Ingestion")
    print(f"📁 Source: {MCC_FOLDER}")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Load contracts
    print("\n📂 Loading contracts...")
    contracts = load_stanford_contracts()
    
    if not contracts:
        return
    
    # Step 2: Split into chunks
    print("\n✂️ Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(contracts)
    print(f"   Created {len(chunks)} chunks from {len(contracts)} contracts")
    
    # Step 3: Create embeddings
    print("\n🔤 Creating embeddings (this will take time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Fast, good quality
        model_kwargs={'device': 'cpu'}
    )
    
    # Step 4: Store in ChromaDB
    print(f"\n💾 Storing in ChromaDB: {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Step 5: Persist
    print("\n💿 Saving to disk...")
    vectordb.persist()
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    # Summary
    print("\n" + "="*60)
    print("✅ STANFORD INGESTION COMPLETE")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Files processed: {len(contracts)}")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Database: {CHROMA_DB_PATH}")
    print(f"   - Collection: {COLLECTION_NAME}")
    print(f"   - Time taken: {minutes}m {seconds}s")
    print("="*60)

if __name__ == "__main__":
    ingest_stanford()