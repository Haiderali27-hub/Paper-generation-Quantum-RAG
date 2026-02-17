import os
import re
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Path to PDFs
PDF_FOLDER = "pdfs"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
embedding_size = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_size)

documents = []
embeddings = []

def clean_text(text):
    """Clean and validate text content"""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '').replace('\ufffd', '')
    
    return text

def is_valid_text(text):
    """Check if text is valid for embedding"""
    if not text or not isinstance(text, str):
        return False
    
    cleaned = clean_text(text)
    return len(cleaned) >= 20  # Minimum length requirement

# Read PDFs from all subfolders
for root, dirs, files in os.walk(PDF_FOLDER):
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)
            print(f"Processing: {path}")
            
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text

                # Clean the full text first
                text = clean_text(text)
                
                if not text:
                    print(f"⚠️ No valid text found in {path}")
                    continue

                # Split into chunks
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for chunk in chunks:
                    chunk = clean_text(chunk)
                    if is_valid_text(chunk):
                        try:
                            emb = model.encode(chunk)
                            index.add(np.array([emb]))
                            documents.append(chunk)
                        except Exception as e:
                            print(f"⚠️ Error encoding chunk: {str(e)[:100]}...")
                            continue
                
                print(f"✅ Successfully processed {path}")
                            
            except Exception as e:
                print(f"❌ Error processing {path}: {str(e)}")
                continue

print(f"\n📊 Total documents indexed: {len(documents)}")

# Simple search function
def search(query, k=3):
    if not query or not isinstance(query, str):
        return []
    
    query = clean_text(query)
    if not is_valid_text(query):
        print("⚠️ Invalid query")
        return []
    
    try:
        q_emb = model.encode(query)
        distances, indices = index.search(np.array([q_emb]), k)
        results = [documents[i] for i in indices[0] if i < len(documents)]
        return results
    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
        return []

# Example usage
if __name__ == "__main__":
    if len(documents) == 0:
        print("❌ No documents were successfully indexed. Check your PDF files.")
    else:
        query = "quantum computing basics"
        results = search(query)
        print(f"\n🔍 Search results for: '{query}'")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(">>", r[:200], "...")
