import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import io
import re
import time
import hashlib
from typing import List, Dict, Tuple
import pandas as pd
import plotly.express as px
import numpy as np

# Configure page
st.set_page_config(
    page_title="Smart RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None

class SmartRAGSystem:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    @st.cache_resource
    def load_embedder():
        """Load embedding model with caching"""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def initialize_chroma(self):
        """Initialize ChromaDB"""
        if st.session_state.chroma_client is None:
            st.session_state.chroma_client = chromadb.Client()
            try:
                st.session_state.collection = st.session_state.chroma_client.create_collection(
                    name="documents",
                    metadata={"description": "Document chunks for RAG"}
                )
            except:
                st.session_state.collection = st.session_state.chroma_client.get_collection("documents")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        return text
    
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        current_page = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for page markers
            if "--- Page" in sentence:
                page_match = re.search(r'--- Page (\d+) ---', sentence)
                if page_match:
                    current_page = int(page_match.group(1))
                continue
            
            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page': current_page,
                        'document': doc_name,
                        'chunk_id': len(chunks)
                    })
                current_chunk = sentence + ". "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': current_page,
                'document': doc_name,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def simple_sentiment_analysis(self, text: str) -> Dict:
        """Simple rule-based sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'effective', 'beneficial']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'ineffective', 'harmful', 'poor', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'POSITIVE', 'score': 0.7 + (positive_count * 0.1)}
        elif negative_count > positive_count:
            return {'label': 'NEGATIVE', 'score': 0.7 + (negative_count * 0.1)}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def extract_simple_entities(self, text: str) -> Dict:
        """Simple entity extraction using regex patterns"""
        entities = {}
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            entities['DATES'] = list(set(dates))
        
        # Extract capitalized words (potential names/organizations)
        name_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
        names = re.findall(name_pattern, text)
        # Filter out common words
        common_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
        names = [name for name in names if name not in common_words and len(name) > 2]
        if names:
            entities['NAMES'] = list(set(names))
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?%?\b'
        numbers = re.findall(number_pattern, text)
        if numbers:
            entities['NUMBERS'] = list(set(numbers))
        
        return entities
    
    def classify_question(self, question: str) -> str:
        """Simple question classification"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'explain', 'describe']):
            return "Factual"
        elif any(word in question_lower for word in ['why', 'how', 'analyze', 'compare']):
            return "Analytical"
        elif any(word in question_lower for word in ['summarize', 'summary', 'overview', 'main']):
            return "Summary"
        else:
            return "General"
    
    def process_document(self, uploaded_file, doc_name: str):
        """Process uploaded document"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract text
        status_text.text("📄 Extracting text from PDF...")
        progress_bar.progress(20)
        
        text = self.extract_text_from_pdf(uploaded_file)
        
        # Create chunks
        status_text.text("✂️ Creating text chunks...")
        progress_bar.progress(40)
        
        chunks = self.chunk_text(text, doc_name)
        
        # Generate embeddings
        status_text.text("🧠 Generating embeddings...")
        progress_bar.progress(60)
        
        embeddings = st.session_state.embedder.encode([chunk['text'] for chunk in chunks])
        
        # Store in ChromaDB
        status_text.text("💾 Storing in vector database...")
        progress_bar.progress(80)
        
        # Prepare data for ChromaDB
        ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{'page': chunk['page'], 'document': chunk['document']} for chunk in chunks]
        
        st.session_state.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Document processed successfully!")
        
        # Store document info
        st.session_state.documents.append({
            'name': doc_name,
            'chunks': len(chunks),
            'text_length': len(text),
            'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        time.sleep(1)  # Show completion
        progress_bar.empty()
        status_text.empty()
    
    def retrieve_relevant_chunks(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks for question"""
        if st.session_state.collection is None:
            return []
        
        # Generate question embedding
        question_embedding = st.session_state.embedder.encode([question])
        
        # Search in ChromaDB
        results = st.session_state.collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=top_k
        )
        
        relevant_chunks = []
        for i in range(len(results['documents'][0])):
            relevant_chunks.append({
                'text': results['documents'][0][i],
                'page': results['metadatas'][0][i]['page'],
                'document': results['metadatas'][0][i]['document'],
                'distance': results['distances'][0][i]
            })
        
        return relevant_chunks
    
    def generate_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer from relevant chunks"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or upload a relevant document."
        
        # Simple extractive approach
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        question_type = self.classify_question(question)
        
        if question_type == "Summary":
            answer = f"Based on the document analysis:\n\n{context[:600]}..."
        elif question_type == "Factual":
            # Look for direct answers
            sentences = context.split('.')
            question_words = question.lower().split()
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_words if len(word) > 3):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                answer = '. '.join(relevant_sentences[:3]) + '.'
            else:
                answer = f"Based on the context: {context[:400]}..."
        else:
            answer = f"According to the document:\n\n{context[:500]}..."
        
        return answer
    
    def ask_question(self, question: str) -> Dict:
        """Main Q&A function"""
        start_time = time.time()
        
        # Classify question
        question_type = self.classify_question(question)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=3)
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Calculate confidence (simple heuristic)
        confidence = 0.2
        if relevant_chunks:
            avg_distance = sum(chunk['distance'] for chunk in relevant_chunks) / len(relevant_chunks)
            confidence = max(0.2, min(0.95, 1.0 - avg_distance))
        
        # Process time
        process_time = time.time() - start_time
        
        # Simple analysis
        sentiment = self.simple_sentiment_analysis(answer)
        entities = self.extract_simple_entities(answer)
        
        result = {
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'confidence': confidence,
            'sources': relevant_chunks,
            'sentiment': sentiment,
            'entities': entities,
            'process_time': process_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store in history
        st.session_state.qa_history.append(result)
        
        return result

# Initialize the system
@st.cache_resource
def get_rag_system():
    return SmartRAGSystem()

# Main app
def main():
    st.title("🤖 Smart RAG Q&A System")
    st.markdown("### Upload documents and ask intelligent questions with AI-powered analysis")
    
    # Initialize system
    rag_system = get_rag_system()
    
    # Load embedding model
    if st.session_state.embedder is None:
        with st.spinner("🚀 Loading AI models... (This may take a minute on first run)"):
            st.session_state.embedder = SmartRAGSystem.load_embedder()
    
    # Initialize ChromaDB
    rag_system.initialize_chroma()
    
    # Sidebar
    st.sidebar.header("📁 Document Management")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        doc_name = uploaded_file.name
        
        if st.sidebar.button("🔄 Process Document"):
            with st.spinner("Processing document..."):
                rag_system.process_document(uploaded_file, doc_name)
                st.sidebar.success(f"✅ Processed: {doc_name}")
                st.rerun()
    
    # Show processed documents
    if st.session_state.documents:
        st.sidebar.subheader("📚 Processed Documents")
        for doc in st.session_state.documents:
            st.sidebar.write(f"📄 **{doc['name']}**")
            st.sidebar.write(f"   - {doc['chunks']} chunks")
            st.sidebar.write(f"   - {doc['text_length']:,} characters")
            st.sidebar.write(f"   - {doc['processed_at']}")
            st.sidebar.write("---")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Sample questions
        if st.session_state.documents:
            st.subheader("🎯 Try these sample questions:")
            sample_questions = [
                "What is the main topic of this document?",
                "Summarize the key findings",
                "What are the important dates mentioned?",
                "What are the main conclusions?",
                "Explain the methodology used"
            ]
            
            selected_question = st.selectbox("Select a sample question:", 
                                           [""] + sample_questions)
        
        # Question input
        question = st.text_input("Or type your own question:", 
                               value=selected_question if 'selected_question' in locals() else "")
        
        # Ask button
        if st.button("🔍 Ask Question", disabled=not question or not st.session_state.documents):
            if not st.session_state.documents:
                st.error("Please upload and process a document first!")
            else:
                with st.spinner("🤔 Analyzing question and generating answer..."):
                    result = rag_system.ask_question(question)
                    
                    # Display answer
                    st.subheader("📝 Answer")
                    st.write(result['answer'])
                    
                    # Display metadata
                    col_conf, col_type, col_time = st.columns(3)
                    with col_conf:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col_type:
                        st.metric("Question Type", result['question_type'])
                    with col_time:
                        st.metric("Process Time", f"{result['process_time']:.2f}s")
                    
                    # Display sources
                    if result['sources']:
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources']):
                            relevance = 1 - source['distance']
                            with st.expander(f"📄 Source {i+1} - Page {source['page']} (Relevance: {relevance:.1%})"):
                                st.write(source['text'])
                    
                    # Display analysis
                    if result['entities'] or result['sentiment']:
                        st.subheader("🔍 AI Analysis")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            if result['entities']:
                                st.write("**🏷️ Key Information Extracted:**")
                                for entity_type, entities in result['entities'].items():
                                    st.write(f"- **{entity_type}**: {', '.join(set(entities))}")
                        
                        with analysis_col2:
                            if result['sentiment']:
                                st.write("**😊 Sentiment Analysis:**")
                                sentiment_color = "green" if result['sentiment']['label'] == 'POSITIVE' else "red" if result['sentiment']['label'] == 'NEGATIVE' else "gray"
                                st.write(f"- **Tone**: :{sentiment_color}[{result['sentiment']['label']}]")
                                st.write(f"- **Confidence**: {result['sentiment']['score']:.3f}")
    
    with col2:
        st.header("📊 Analytics Dashboard")
        
        # Document stats
        if st.session_state.documents:
            st.subheader("📈 Document Statistics")
            
            # Create dataframe for visualization
            df = pd.DataFrame(st.session_state.documents)
            
            # Chunks chart
            fig_chunks = px.bar(df, x='name', y='chunks', 
                              title='Chunks per Document',
                              labels={'name': 'Document', 'chunks': 'Number of Chunks'})
            fig_chunks.update_layout(height=300)
            st.plotly_chart(fig_chunks, use_container_width=True)
            
            # Character count
            fig_chars = px.bar(df, x='name', y='text_length',
                             title='Document Length (Characters)',
                             labels={'name': 'Document', 'text_length': 'Characters'})
            fig_chars.update_layout(height=300)
            st.plotly_chart(fig_chars, use_container_width=True)
        
        # Q&A History
        if st.session_state.qa_history:
            st.subheader("🕒 Recent Q&A History")
            
            # Recent questions
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
                with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:30]}..."):
                    st.write(f"**❓ Question:** {qa['question']}")
                    st.write(f"**💡 Answer:** {qa['answer'][:150]}...")
                    st.write(f"**🎯 Confidence:** {qa['confidence']:.1%}")
                    st.write(f"**⏰ Time:** {qa['timestamp']}")
            
            # Question type distribution
            if len(st.session_state.qa_history) > 1:
                question_types = [qa['question_type'] for qa in st.session_state.qa_history]
                type_counts = pd.Series(question_types).value_counts()
                
                fig_types = px.pie(values=type_counts.values, names=type_counts.index,
                                 title='Question Types Distribution')
                fig_types.update_layout(height=300)
                st.plotly_chart(fig_types, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 **Built with Modern AI Stack**")
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown("**🤖 AI Models:**")
        st.markdown("- Sentence-BERT Embeddings")
        st.markdown("- Vector Similarity Search")
        st.markdown("- Smart Text Chunking")
    
    with col_tech2:
        st.markdown("**⚡ Tech Stack:**")
        st.markdown("- Streamlit Frontend")
        st.markdown("- ChromaDB Vector Store")
        st.markdown("- PyTorch Backend")
    
    with col_tech3:
        st.markdown("**🎯 Features:**")
        st.markdown("- Real-time Processing")
        st.markdown("- Source Attribution")
        st.markdown("- Analytics Dashboard")

if __name__ == "__main__":
    main()