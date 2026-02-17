import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
DB_DIR = "faiss_index"  
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
# ----------------------------------------

def main():
    print("🔍 Loading FAISS DB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ Database loaded successfully.")

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower().strip() == "exit":
            print("👋 Goodbye!")
            break

        docs = db.similarity_search(query, k=TOP_K)

        print("\n📌 Retrieved Context:")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Result {i} ---")
            print(f"📂 Source: {doc.metadata.get('source', 'unknown')}")
            print(doc.page_content[:600] + "...")


if __name__ == "__main__":
    main()
