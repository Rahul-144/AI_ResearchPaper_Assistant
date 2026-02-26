import streamlit as st
from rag_engine import RAG_Engine
from Faiss_index import faiss_index
from parser import load_pdf, split_into_sections
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
    }
    .answer-box {
        background-color: #ffffff; /* use white background for higher contrast */
        color: #000000;            /* dark text for readability */
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .chunk-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin-bottom: 1rem;
    }
    .section-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📄 AI Research Paper Assistant")
st.markdown("""
Browse and query the research paper with AI-powered search and answers.
This tool uses semantic search, cross-encoder reranking, and LLM-based answer generation.
""")

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    
    st.divider()
    st.subheader("📤 Upload Paper")
    uploaded_file = st.file_uploader(
        "Upload a research paper (PDF):",
        type="pdf",
        help="Upload a PDF file to analyze"
    )
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        st.text(f"File: {uploaded_file.name}")
    else:
        st.warning("Please upload a PDF to enable the assistant.")
    
    # radio navigation moves outside of conditional block so it always executes
    if uploaded_file is not None:
        page = st.radio("Select a page:", 
                        ["🤖 Ask Question", "📑 Browse Sections", "📊 Paper Overview"])
    else:
        page = None
    
    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    This assistant analyzes research papers using:
    - **FAISS Vector Store** for semantic search
    - **Cross-Encoder Reranking** for relevance filtering
    - **LLM (Groq API)** for answer generation
    """)

# Load data (cached for performance)
@st.cache_resource
def get_vectorstore_and_sections(file_path):
    """Load and cache the FAISS vector store and sections from a given PDF path"""
    vectorstore = faiss_index()
    pages = load_pdf(file_path)
    sections = split_into_sections(pages)
    return vectorstore, sections

# initialize
vectorstore = None
sections = []

# Determine which file to use and load when available
if uploaded_file is not None:
    import tempfile
    import os
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    # clear cached result to force rebuild when file changes
    try:
        get_vectorstore_and_sections.clear()
    except Exception:
        pass
    vectorstore, sections = get_vectorstore_and_sections(temp_path)


# Main content area
if uploaded_file is None or page is None:
    st.info("Upload a PDF file from the sidebar to enable search and browsing features.")
else:
    # Page 1: Ask Question
    if page == "🤖 Ask Question":
        st.header("Ask a Question")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your question about the paper:",
              
            )
        
        with col2:
            search_button = st.button("🔍 Search", use_container_width=True)
        
        if search_button and query:
            with st.spinner("Searching and generating answer..."):
                try:
                    # Generate answer using RAG
                    answer = RAG_Engine(query)
                    
                    # Display answer
                    st.markdown("### 📌 Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Retrieve and display source chunks
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
                    relevant_chunks = retriever.invoke(query)
                    
                    st.markdown("### 📚 Source Chunks")
                    st.markdown(f"*Found {len(relevant_chunks)} relevant chunks from the paper*")
                    
                    for idx, chunk in enumerate(relevant_chunks, 1):
                        section = chunk.metadata.get('section', 'Unknown')
                        heading = chunk.metadata.get('heading', section)
                        
                        with st.expander(f"📖 Chunk {idx} - {heading}", expanded=(idx <= 2)):
                            st.markdown(f'<span class="section-tag">{section}</span>', 
                                      unsafe_allow_html=True)
                            st.markdown(chunk.page_content)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

    # Page 2: Browse Sections
    elif page == "📑 Browse Sections":
        st.header("Browse Paper Sections")
        
        # Create a select box with all sections
        section_titles = [heading for heading, _ in sections]
        selected_section = st.selectbox(
            "Select a section:",
            section_titles,
            index=0
        )
        
        # Find and display the selected section
        selected_idx = section_titles.index(selected_section)
        heading, content = sections[selected_idx]
        
        st.markdown(f"### {heading}")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Section Number", heading.split()[0] if heading[0].isdigit() else "N/A")
        with col2:
            st.metric("Content Length", f"{len(content)} chars")
        with col3:
            st.metric("Word Count", len(content.split()))
        
        st.divider()
        
        # Display content in a scrollable container using text area
        st.text_area(
            "Section Content:",
            value=content,
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if selected_idx > 0:
                if st.button("⬅️ Previous Section"):
                    st.session_state.selected_idx = selected_idx - 1
                    st.rerun()
        
        with col3:
            if selected_idx < len(sections) - 1:
                if st.button("Next Section ➡️"):
                    st.session_state.selected_idx = selected_idx + 1
                    st.rerun()

    # Page 3: Paper Overview
    elif page == "📊 Paper Overview":
        st.header("Paper Overview")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sections", len(sections))
        
        with col2:
            total_chars = sum(len(content) for _, content in sections)
            st.metric("Total Characters", f"{total_chars:,}")
        
        with col3:
            total_words = sum(len(content.split()) for _, content in sections)
            st.metric("Total Words", f"{total_words:,}")
        
        with col4:
            st.metric("Vector Store Size", vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else "N/A")
        
        st.divider()
        
        # Section list with statistics
        st.subheader("📚 Sections Breakdown")
        
        section_data = []
        for idx, (heading, content) in enumerate(sections, 1):
            section_data.append({
                "Section": heading,
                "Characters": len(content),
                "Words": len(content.split()),
                "Paragraphs": len([p for p in content.split('\n') if p.strip()])
            })
        
        # Display as a table
        df = pd.DataFrame(section_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        st.subheader("📊 Content Statistics")
        
        avg_section_size = df['Characters'].mean()
        largest_section = df.loc[df['Characters'].idxmax()]
        smallest_section = df.loc[df['Characters'].idxmin()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Avg Section Size**: {int(avg_section_size):,} chars")
        with col2:
            st.success(f"**Largest**: {largest_section['Section']}")
        with col3:
            st.warning(f"**Smallest**: {smallest_section['Section']}")
# Footer
st.divider()
st.markdown("""
---
*Built with Streamlit | Powered by FAISS, HuggingFace, and OpenAI-compatible APIs*
""")
