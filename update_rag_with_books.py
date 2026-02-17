"""
Update RAG Database with Books Folder
This will add the pdfs/books folder to your existing FAISS database
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
BOOKS_DIR = "pdfs/books"
DB_DIR = "faiss_index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TEXT_LENGTH = 50


def clean_text(text):
    """Clean and validate text content"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.replace('\x00', '').replace('\ufffd', '')
    return text


def load_books(books_dir):
    """Load PDFs from books folder"""
    docs = []
    success_count = 0
    error_count = 0
    
    for root, _, files in os.walk(books_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                path = os.path.join(root, fname)
                print(f"📚 Loading {fname} ...")
                
                try:
                    loader = PyPDFLoader(path)
                    pdf_docs = loader.load()
                    
                    valid_docs = []
                    for doc in pdf_docs:
                        cleaned_content = clean_text(doc.page_content)
                        if len(cleaned_content) >= MIN_TEXT_LENGTH:
                            doc.page_content = cleaned_content
                            valid_docs.append(doc)
                    
                    if valid_docs:
                        docs.extend(valid_docs)
                        success_count += 1
                        print(f"✅ Loaded {len(valid_docs)} pages from {fname}")
                    else:
                        print(f"⚠️  No valid content in {fname}")
                        error_count += 1
                        
                except Exception as e:
                    print(f"❌ Error loading {fname}: {str(e)}")
                    error_count += 1
                    continue
    
    print(f"\n📊 Summary: {success_count} books loaded, {error_count} failed")
    return docs


def update_database():
    """Update existing FAISS database with new books"""
    print("📥 Loading books from pdfs/books folder...")
    documents = load_books(BOOKS_DIR)
    
    if not documents:
        print("❌ No books loaded.")
        return

    print(f"📚 Loaded {len(documents)} pages from books")

    print("✂️  Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    print(f"📝 Created {len(texts)} text chunks")

    print("🧠 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    print("💾 Updating FAISS database...")
    try:
        # Load existing database
        if os.path.exists(DB_DIR):
            print("📂 Loading existing database...")
            db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            old_count = db.index.ntotal
            print(f"   Current vectors: {old_count}")
            
            # Add new documents in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"   Adding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                try:
                    batch_db = FAISS.from_documents(batch, embeddings)
                    db.merge_from(batch_db)
                except Exception as e:
                    print(f"⚠️  Error in batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            # Save updated database
            db.save_local(DB_DIR)
            new_count = db.index.ntotal
            print(f"\n✅ Database updated successfully!")
            print(f"📊 Old vectors: {old_count}")
            print(f"📊 New vectors: {new_count}")
            print(f"📊 Added: {new_count - old_count} vectors")
            
        else:
            print("❌ No existing database found. Run ingest.py first!")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🔄 RAG Database Updater - Adding Books")
    print("=" * 60)
    update_database()
    print("\n🎉 Done! You can now query both quantum and finance topics!")
    print("Run: python multi_domain_query.py")