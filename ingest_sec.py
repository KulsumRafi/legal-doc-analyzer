"""
SEC EDGAR Live Data Ingestion
Fetches recent contracts via SEC API and stores in ChromaDB
"""

import os
import time
import requests
from sec_api import QueryApi
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from datetime import datetime, timedelta

# 🔑 SEC API KEY - LOAD FROM ENVIRONMENT
SEC_API_KEY = os.environ.get("SEC_API_KEY")
if not SEC_API_KEY:
    raise ValueError("❌ SEC_API_KEY not found! Set it in environment variables.")

# Configuration
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "sec_live"
DAYS_TO_FETCH = 30
MAX_FILINGS = 20

def fetch_sec_filings():
    """Fetch recent material contracts from SEC"""
    
    queryApi = QueryApi(api_key=SEC_API_KEY)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"formType": "8-K"}},
                    {"match": {"description": "EX-10"}}
                ],
                "filter": [
                    {
                        "range": {
                            "filedAt": {
                                "gte": start_date.strftime("%Y-%m-%d"),
                                "lte": end_date.strftime("%Y-%m-%d")
                            }
                        }
                    }
                ]
            }
        },
        "from": "0",
        "size": str(MAX_FILINGS),
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    print(f"🔍 Querying SEC API for contracts from last {DAYS_TO_FETCH} days...")
    response = queryApi.get_filings(query)
    
    filings = response.get('filings', [])
    print(f"✅ Found {len(filings)} filings")
    
    return filings

def download_contract_text(filing):
    """Download and parse contract text"""
    
    # Find exhibit URL
    exhibit_url = None
    for doc in filing.get('documentFormatFiles', []):
        if doc.get('type', '').startswith('EX-10'):
            exhibit_url = doc.get('documentUrl')
            break
    
    if not exhibit_url:
        return "", None
    
    metadata = {
        "source": exhibit_url,
        "source_type": "sec_live",
        "ticker": filing.get('ticker', 'Unknown'),
        "company": filing.get('companyName', 'Unknown'),
        "filed_at": filing.get('filedAt', '')[:10],
        "form_type": filing.get('formType', '')
    }
    
    try:
        headers = {'User-Agent': 'Legal Document Analyzer (your@email.com)'}
        response = requests.get(exhibit_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up
            text = ' '.join(text.split())
            
            # Truncate if too long
            if len(text) > 50000:
                text = text[:50000] + "... [truncated]"
            
            return text, metadata
    
    except Exception as e:
        print(f"   ❌ Download error: {e}")
    
    return "", None

def ingest_sec():
    """Main SEC ingestion function"""
    
    print("="*60)
    print("🟢 SEC Live Data Ingestion")
    print("="*60)
    
    # Step 1: Fetch filings
    print("\n📋 Fetching filings...")
    filings = fetch_sec_filings()
    
    if not filings:
        print("❌ No filings found")
        return
    
    # Step 2: Initialize components
    print("\n🔤 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Step 3: Initialize ChromaDB (CREATES folder if needed)
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Step 4: Process each filing
    contracts_added = 0
    chunks_added = 0
    
    print(f"\n📥 Processing {len(filings)} filings...")
    
    for i, filing in enumerate(filings, 1):
        print(f"\n[{i}/{len(filings)}] {filing.get('ticker', 'N/A')} - {filing.get('filedAt', '')[:10]}")
        
        text, metadata = download_contract_text(filing)
        
        if text and metadata:
            doc = Document(page_content=text, metadata=metadata)
            chunks = text_splitter.split_documents([doc])
            
            if chunks:
                vectordb.add_documents(chunks)
                vectordb.persist()
                
                contracts_added += 1
                chunks_added += len(chunks)
                print(f"   ✅ Added {len(chunks)} chunks")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("✅ SEC INGESTION COMPLETE")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Contracts added: {contracts_added}")
    print(f"   - Total chunks: {chunks_added}")
    print(f"   - Database: {CHROMA_DB_PATH}")
    print(f"   - Collection: {COLLECTION_NAME}")
    print("="*60)

if __name__ == "__main__":
    ingest_sec()