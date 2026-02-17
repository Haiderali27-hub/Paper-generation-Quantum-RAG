"""
Direct Quantum Operating System Kernels Paper Generator
Generates research paper with mathematical equations
"""

import os
import requests
import json
import datetime
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def test_ollama():
    """Test Ollama connection"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Hello! Write a simple quantum equation.",
                "stream": False,
                "options": {"num_predict": 100}
            },
            timeout=30
        )
        return response.status_code == 200
    except:
        return False


def load_knowledge_base():
    """Load the FAISS knowledge base"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("✅ Knowledge base loaded successfully")
        return db
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        return None


def get_context(db, query, k=5):
    """Get relevant context from knowledge base"""
    try:
        docs = db.similarity_search(query, k=k)
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
        return context
    except:
        return ""


def query_ollama(prompt):
    """Query Ollama model"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2000
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"[Error: {response.status_code}]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def format_equations(text):
    """Format mathematical expressions as LaTeX equations"""
    # Look for mathematical patterns and format them
    patterns = [
        (r'([A-Z])\s*=\s*([^,\.\n]{3,50})', r'$$\1 = \2$$'),  # Simple equations
        (r'\|([^|]+)\⟩', r'$$|\1⟩$$'),  # Quantum states
        (r'⟨([^⟨]+)\|', r'$$⟨\1|$$'),   # Bra states
        (r'([A-Z])†([A-Z])', r'$$\1^†\2$$'),  # Hermitian conjugate
        (r'∑([^,\.\n]{3,30})', r'$$\\sum \1$$'),  # Summations
        (r'∏([^,\.\n]{3,30})', r'$$\\prod \1$$'),  # Products
    ]
    
    formatted_text = text
    for pattern, replacement in patterns:
        formatted_text = re.sub(pattern, replacement, formatted_text)
    
    return formatted_text


def generate_section(db, topic, section_name, requirements):
    """Generate a single section with equations"""
    print(f"📝 Generating {section_name}...")
    
    # Get context
    context_query = f"{topic} {section_name}"
    context = get_context(db, context_query, k=6)
    
    # Special math instructions for certain sections
    math_instruction = ""
    if any(word in section_name.lower() for word in ["mathematical", "framework", "implementation", "analysis"]):
        math_instruction = """
MATHEMATICAL REQUIREMENTS:
- Include specific equations and formulas
- Use quantum computing notation: |ψ⟩, ⟨φ|, Ĥ, Û, ρ
- Show mathematical relationships
- Include scheduling algorithms and complexity analysis
- Use proper mathematical symbols and operators
"""
    
    prompt = f"""Write a detailed {section_name} section for a research paper on "{topic}".

CONTEXT FROM RESEARCH PAPERS:
{context}

SECTION REQUIREMENTS:
{requirements}

{math_instruction}

FORMATTING INSTRUCTIONS:
1. Write 600-1000 words
2. Use academic tone and technical language
3. Include mathematical expressions using proper notation
4. For quantum states use: |ψ⟩, |0⟩, |1⟩
5. For operators use: Ĥ (Hamiltonian), Û (Unitary), ρ̂ (density matrix)
6. Include scheduling equations and complexity analysis
7. Mark figure placeholders as: [**FIGURE X: Description**]
8. Include citations as: [Ref-X]
9. Use subsections with ### headers where appropriate

Write the complete {section_name} section:"""

    response = query_ollama(prompt)
    
    # Format equations
    formatted_response = format_equations(response)
    
    return formatted_response


def generate_qos_paper():
    """Generate the complete Quantum Operating System Kernels paper"""
    
    topic = "Quantum Operating System Kernels: Process Isolation and Qubit Scheduling Architectures"
    
    print(f"🚀 Generating research paper: {topic}")
    print("=" * 80)
    
    # Test Ollama
    if not test_ollama():
        print("❌ Ollama not responding. Please check if it's running.")
        return None
    
    # Load knowledge base
    db = load_knowledge_base()
    if not db:
        print("❌ Could not load knowledge base")
        return None
    
    # Paper sections with specific requirements
    sections = {
        "Introduction": """
        - Define quantum operating systems and their necessity
        - Explain the challenges of qubit scheduling and process isolation
        - Outline the paper's contributions to quantum OS design
        - Discuss current limitations in quantum computing infrastructure
        - Present the research objectives and methodology
        """,
        
        "Background and Related Work": """
        - Review existing quantum computing frameworks
        - Analyze classical OS concepts applied to quantum systems
        - Discuss QOS, QuDOS, and other prototype quantum operating systems
        - Compare quantum vs classical process management
        - Identify gaps in current quantum OS research
        """,
        
        "Mathematical Framework for Quantum Process Management": """
        - Define mathematical models for qubit allocation
        - Present scheduling algorithms with complexity analysis
        - Formulate process isolation using quantum information theory
        - Include Hilbert space partitioning for process separation
        - Develop mathematical models for quantum resource management
        """,
        
        "Quantum Scheduling Algorithms": """
        - Present novel qubit scheduling algorithms
        - Analyze time complexity and space requirements
        - Include mathematical proofs of algorithm correctness
        - Compare different scheduling strategies
        - Discuss quantum circuit optimization within scheduling
        """,
        
        "Process Isolation Architecture": """
        - Design quantum process isolation mechanisms
        - Present security models for quantum processes
        - Include mathematical analysis of isolation guarantees
        - Discuss quantum error propagation between processes
        - Analyze overhead costs of isolation
        """,
        
        "Implementation and Performance Analysis": """
        - Present implementation details and system architecture
        - Analyze performance metrics and benchmarks
        - Include mathematical models for system performance
        - Discuss scalability and resource utilization
        - Compare with existing quantum computing platforms
        """,
        
        "Results and Evaluation": """
        - Present experimental results and analysis
        - Include performance comparisons and metrics
        - Analyze system efficiency and resource utilization
        - Discuss limitations and trade-offs
        - Validate theoretical models with experimental data
        """,
        
        "Discussion and Future Work": """
        - Discuss implications for quantum computing infrastructure
        - Analyze practical deployment challenges
        - Identify future research directions
        - Discuss standardization needs for quantum operating systems
        - Present roadmap for quantum OS development
        """,
        
        "Conclusion": """
        - Summarize key contributions and findings
        - Highlight the significance for quantum computing
        - Discuss practical implications and applications
        - Present final thoughts on quantum OS evolution
        """
    }
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quantum_os_kernels_with_equations_{timestamp}.md"
    
    # Start paper content
    paper_content = f"""# {topic}

**Authors:** [Author Names]  
**Affiliation:** [Institution/Affiliation]  
**Date:** {datetime.datetime.now().strftime("%Y-%m-%d")}  

---

## Abstract

[**PLACEHOLDER FOR ABSTRACT - Write a 250-300 word summary after completing all sections**]

**Keywords:** Quantum Operating Systems, Qubit Scheduling, Process Isolation, Quantum Resource Management, Quantum Computing Architecture

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background and Related Work](#background-and-related-work)
3. [Mathematical Framework for Quantum Process Management](#mathematical-framework-for-quantum-process-management)
4. [Quantum Scheduling Algorithms](#quantum-scheduling-algorithms)
5. [Process Isolation Architecture](#process-isolation-architecture)
6. [Implementation and Performance Analysis](#implementation-and-performance-analysis)
7. [Results and Evaluation](#results-and-evaluation)
8. [Discussion and Future Work](#discussion-and-future-work)
9. [Conclusion](#conclusion)
10. [References](#references)

---

"""
    
    # Generate each section
    for i, (section_name, requirements) in enumerate(sections.items(), 1):
        print(f"📖 Section {i}/{len(sections)}: {section_name}")
        
        section_content = generate_section(db, topic, section_name, requirements)
        paper_content += f"\n## {section_name}\n\n{section_content}\n\n"
        
        print(f"✅ Completed ({len(section_content)} characters)")
    
    # Add references and appendices
    paper_content += """
## References

[**PLACEHOLDER FOR REFERENCES**]

*Instructions for completing references:*
1. Replace [Ref-X] citations with actual references
2. Use IEEE format: [1] Author, "Title," Journal, vol. X, no. Y, pp. Z-W, Year.
3. Include 20-30 high-quality references
4. Focus on quantum computing, operating systems, and scheduling literature

---

## Appendices

### Appendix A: Mathematical Proofs and Derivations
[**PLACEHOLDER FOR DETAILED MATHEMATICAL PROOFS**]

### Appendix B: Algorithm Implementations
[**PLACEHOLDER FOR PSEUDOCODE AND IMPLEMENTATION DETAILS**]

### Appendix C: Performance Benchmarks
[**PLACEHOLDER FOR DETAILED PERFORMANCE DATA**]

---

**Paper Statistics:**
- Sections: 9
- Mathematical expressions: [Count $$...$$]
- Figures: [Count [**FIGURE**]]
- References: [Count [Ref-X]]

**LaTeX Conversion Notes:**
- All equations marked with $$ are ready for LaTeX
- Replace figure placeholders with actual diagrams
- Verify all mathematical notation for accuracy
"""
    
    # Save the paper
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(paper_content)
        
        # Count elements
        equation_count = paper_content.count("$$")
        figure_count = paper_content.count("[**FIGURE")
        ref_count = paper_content.count("[Ref-")
        
        print(f"\n🎉 Research paper generated successfully!")
        print(f"📄 File: {filename}")
        print(f"📊 Statistics:")
        print(f"   - Total length: {len(paper_content):,} characters")
        print(f"   - Mathematical expressions: {equation_count}")
        print(f"   - Figure placeholders: {figure_count}")
        print(f"   - Citation placeholders: {ref_count}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error saving paper: {e}")
        return None


if __name__ == "__main__":
    print("🌌 Quantum Operating System Kernels Paper Generator")
    print("📚 With Mathematical Equations Support")
    print("=" * 60)
    
    filename = generate_qos_paper()
    
    if filename:
        print(f"\n🏆 SUCCESS!")
        print(f"📂 Next steps:")
        print(f"   1. Review and refine mathematical equations")
        print(f"   2. Create system architecture diagrams")
        print(f"   3. Complete the reference list")
        print(f"   4. Write the abstract")
        print(f"   5. Convert equations to proper LaTeX format")
    else:
        print("❌ Paper generation failed")