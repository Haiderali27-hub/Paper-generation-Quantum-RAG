import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- CONFIG ----------------
PDF_DIR = "pdfs"        # folder where all your PDFs are (with subfolders)
DB_DIR = "faiss_index"  # where FAISS DB will be stored
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TEXT_LENGTH = 50    # minimum text length to consider valid
# ----------------------------------------


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


def load_pdfs(pdf_dir):
    """Recursively load all PDFs from folder + subfolders with error handling"""
    docs = []
    success_count = 0
    error_count = 0
    
    for root, _, files in os.walk(pdf_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                path = os.path.join(root, fname)
                print(f"📄 Loading {path} ...")
                
                try:
                    loader = PyPDFLoader(path)
                    pdf_docs = loader.load()
                    
                    # Filter and clean documents
                    valid_docs = []
                    for doc in pdf_docs:
                        cleaned_content = clean_text(doc.page_content)
                        if len(cleaned_content) >= MIN_TEXT_LENGTH:
                            doc.page_content = cleaned_content
                            valid_docs.append(doc)
                    
                    if valid_docs:
                        docs.extend(valid_docs)
                        success_count += 1
                        print(f"✅ Successfully loaded {len(valid_docs)} pages")
                    else:
                        print(f"⚠️ No valid content found in {path}")
                        error_count += 1
                        
                except Exception as e:
                    print(f"❌ Error loading {path}: {str(e)}")
                    error_count += 1
                    continue
    
    print(f"\n📊 Summary: {success_count} PDFs loaded successfully, {error_count} failed")
    return docs


def filter_valid_texts(texts):
    """Filter out invalid text chunks that might cause tokenization errors"""
    valid_texts = []
    
    for doc in texts:
        try:
            # Ensure we have valid content
            if (hasattr(doc, 'page_content') and 
                isinstance(doc.page_content, str) and 
                len(doc.page_content.strip()) >= MIN_TEXT_LENGTH):
                
                # Additional cleaning
                doc.page_content = clean_text(doc.page_content)
                valid_texts.append(doc)
                
        except Exception as e:
            print(f"⚠️ Skipping invalid text chunk: {str(e)}")
            continue
    
    return valid_texts


def main():
    print("📥 Reading PDFs...")
    documents = load_pdfs(PDF_DIR)
    
    if not documents:
        print("❌ No documents loaded. Please check your PDF files.")
        return

    print(f"📚 Loaded {len(documents)} document pages")

    print("✂️ Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    
    print("� Filtering valid texts...")
    texts = filter_valid_texts(texts)
    
    if not texts:
        print("❌ No valid text chunks after filtering. Check your PDF content.")
        return
    
    print(f"📝 Processing {len(texts)} text chunks")

    print("�🧠 Loading embeddings model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    except Exception as e:
        print(f"❌ Error loading embeddings model: {str(e)}")
        return

    print("💾 Creating FAISS vector store...")
    try:
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                if i == 0:
                    # Create initial database with first batch
                    db = FAISS.from_documents(batch, embeddings)
                else:
                    # Add subsequent batches
                    batch_db = FAISS.from_documents(batch, embeddings)
                    db.merge_from(batch_db)
            except Exception as e:
                print(f"⚠️ Error processing batch {i//batch_size + 1}: {str(e)}")
                continue

        db.save_local(DB_DIR)
        print(f"✅ Ingestion completed. Database saved to {DB_DIR}")
        print(f"📊 Total vectors created: {db.index.ntotal}")
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return


if __name__ == "__main__":
    main()
