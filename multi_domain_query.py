"""
Multi-Domain RAG Query Tool
Query quantum computing PDFs, finance books, or both combined
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DB_DIR = "faiss_index"  
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def query_rag(query, k=5, filter_domain=None):
    """
    Query the RAG system with optional domain filtering
    
    Args:
        query: Your question
        k: Number of results to return
        filter_domain: 'quantum', 'finance', 'books', or None for all
    """
    print(f"🔍 Searching for: '{query}'")
    if filter_domain:
        print(f"📂 Filtering by domain: {filter_domain}")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
        db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        
        # Get results
        docs = db.similarity_search(query, k=k*2 if filter_domain else k)
        
        # Filter by domain if specified
        filtered_docs = []
        if filter_domain:
            for doc in docs:
                source = doc.metadata.get('source', '').lower()
                
                if filter_domain == 'quantum' and 'books' not in source:
                    filtered_docs.append(doc)
                elif filter_domain == 'finance' and 'books' in source:
                    filtered_docs.append(doc)
                elif filter_domain == 'books' and 'books' in source:
                    filtered_docs.append(doc)
                
                if len(filtered_docs) >= k:
                    break
        else:
            filtered_docs = docs[:k]
        
        print(f"\n📌 Found {len(filtered_docs)} relevant results:\n")
        
        for i, doc in enumerate(filtered_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            # Determine domain
            if 'books' in source.lower():
                domain = "📚 [FINANCE/BOOKS]"
            else:
                domain = "⚛️ [QUANTUM]"
            
            print(f"\n{'='*80}")
            print(f"Result {i} {domain}")
            print(f"Source: {source}")
            print(f"{'='*80}")
            print(doc.page_content[:500] + "...")
            
        return filtered_docs
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def interactive_mode():
    """Interactive query mode with domain selection"""
    print("=" * 60)
    print("🔬 Multi-Domain RAG Query System")
    print("=" * 60)
    print("Domains:")
    print("  ⚛️  Quantum Computing (default PDFs)")
    print("  📚 Finance/Books (pdfs/books folder)")
    print("  🌐 All domains combined")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        query = input("❓ Enter your question (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n🎯 Select domain:")
        print("  1 - Quantum Computing only")
        print("  2 - Finance/Books only")
        print("  3 - All domains (default)")
        
        domain_choice = input("Select (1/2/3 or press Enter for all): ").strip()
        
        domain_map = {
            '1': 'quantum',
            '2': 'books',
            '3': None,
            '': None
        }
        
        filter_domain = domain_map.get(domain_choice, None)
        
        results = query_rag(query, k=5, filter_domain=filter_domain)
        
        if not results:
            print("⚠️  No results found. Try a different query.")


def main():
    """Main function - run interactive mode"""
    interactive_mode()


if __name__ == "__main__":
    main()