import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
import time
import hashlib
from typing import List, Dict, Tuple
import pandas as pd
import plotly.express as px
import numpy as np

# Try to import advanced NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
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
    page_title="Advanced RAG Q&A System",
    page_icon="🧠",
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
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
if 'chunks_data' not in st.session_state:
    st.session_state.chunks_data = []
if 'embeddings_matrix' not in st.session_state:
    st.session_state.embeddings_matrix = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

class AdvancedRAGSystem:
    def __init__(self):
        self.target_chunk_tokens = 100  # Optimal size for embeddings
        self.max_chunk_tokens = 200     # Maximum chunk size
        self.overlap_tokens = 20        # Overlap between chunks
        self.min_chunk_tokens = 30      # Minimum viable chunk
    
    @st.cache_resource
    def load_models():
        """Load all NLP models with caching and error handling"""
        try:
            # Load embedding model (this should always work)
            embedder = SentenceTransformer('all-mpnet-base-v2')
            
            # Try to load SpaCy model
            nlp = None
            if SPACY_AVAILABLE:
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    st.warning("SpaCy model not found. Using fallback NLP processing.")
                    nlp = None
            
            # Try to setup NLTK
            if NLTK_AVAILABLE:
                try:
                    # Try to download NLTK data if not available
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        st.info("Downloading NLTK data... (one-time setup)")
                        nltk.download('punkt', quiet=True)
                        nltk.download('stopwords', quiet=True)
                except Exception as e:
                    st.warning(f"NLTK setup failed: {e}")
            
            return embedder, nlp, True
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, False
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF with better formatting"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                
                # Clean up the text
                page_text = re.sub(r'\n\s*\n', '\n\n', page_text)  # Normalize paragraph breaks
                page_text = re.sub(r'\n(?=[a-z])', ' ', page_text)  # Join broken lines
                
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            return text
            
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""
    
    def advanced_chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """Advanced NLP-based text chunking"""
        chunks = []
        current_page = 1
        
        # Split text by pages first
        pages = re.split(r'--- Page (\d+) ---', text)
        
        for i in range(1, len(pages), 2):  # Skip empty splits
            if i + 1 < len(pages):
                page_num = int(pages[i])
                page_text = pages[i + 1].strip()
                
                if not page_text:
                    continue
                
                # Process this page
                page_chunks = self._chunk_page_text(page_text, page_num, doc_name)
                chunks.extend(page_chunks)
        
        return chunks
    
    def _chunk_page_text(self, text: str, page_num: int, doc_name: str) -> List[Dict]:
        """Chunk text from a single page using advanced NLP or fallback"""
        chunks = []
        
        # Use SpaCy for better sentence segmentation if available
        if st.session_state.nlp and SPACY_AVAILABLE:
            doc = st.session_state.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        elif NLTK_AVAILABLE:
            # Use NLTK if available
            try:
                sentences = sent_tokenize(text)
            except:
                # Fallback to simple splitting
                sentences = self._simple_sentence_split(text)
        else:
            # Fallback to simple splitting
            sentences = self._simple_sentence_split(text)
        
        if not sentences:
            return chunks
        
        # Group sentences into semantic chunks
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            # Calculate tokens (use NLTK if available, otherwise approximate)
            if NLTK_AVAILABLE:
                try:
                    sentence_tokens = len(word_tokenize(sentence))
                except:
                    sentence_tokens = len(sentence.split())
            else:
                sentence_tokens = len(sentence.split())
            
            # Check if adding this sentence exceeds our target
            if (current_chunk_tokens + sentence_tokens > self.max_chunk_tokens and 
                current_chunk_sentences and 
                current_chunk_tokens >= self.min_chunk_tokens):
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(self._create_chunk(chunk_text, page_num, doc_name, len(chunks)))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk_sentences[-self._calculate_overlap_sentences(current_chunk_sentences):]
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_tokens = sum(self._count_tokens(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences and current_chunk_tokens >= self.min_chunk_tokens:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(self._create_chunk(chunk_text, page_num, doc_name, len(chunks)))
        
        return chunks
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting when NLTK is not available"""
        # Simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens with fallback if NLTK not available"""
        if NLTK_AVAILABLE:
            try:
                return len(word_tokenize(text))
            except:
                return len(text.split())
        else:
            return len(text.split())
    
    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        """Calculate optimal overlap in sentences"""
        if len(sentences) <= 2:
            return 0
        elif len(sentences) <= 4:
            return 1
        else:
            return 2
    
    def _create_chunk(self, text: str, page_num: int, doc_name: str, chunk_id: int) -> Dict:
        """Create a standardized chunk object"""
        token_count = self._count_tokens(text)
        
        # Extract key information using NLP (with fallbacks)
        entities = self._extract_chunk_entities(text)
        summary = self._create_chunk_summary(text)
        
        return {
            'text': text,
            'page': page_num,
            'document': doc_name,
            'chunk_id': chunk_id,
            'token_count': token_count,
            'char_count': len(text),
            'sentence_count': len(self._get_sentences(text)),
            'entities': entities,
            'summary': summary
        }
    
    def _get_sentences(self, text: str) -> List[str]:
        """Get sentences with fallback"""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                return self._simple_sentence_split(text)
        else:
            return self._simple_sentence_split(text)
    
    def _extract_chunk_entities(self, text: str) -> Dict:
        """Extract entities from chunk using SpaCy"""
        if not st.session_state.nlp:
            return {}
        
        doc = st.session_state.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def _create_chunk_summary(self, text: str) -> str:
        """Create a brief summary of the chunk"""
        sentences = self._get_sentences(text)
        if len(sentences) <= 2:
            return text[:100] + "..." if len(text) > 100 else text
        else:
            # Return first sentence as summary
            return sentences[0]
    
    def enhanced_sentiment_analysis(self, text: str) -> Dict:
        """Enhanced sentiment analysis using SpaCy"""
        if not st.session_state.nlp:
            return self.simple_sentiment_analysis(text)
        
        doc = st.session_state.nlp(text)
        
        # Analyze sentiment using linguistic features
        positive_indicators = 0
        negative_indicators = 0
        
        for token in doc:
            if token.pos_ == "ADJ":  # Adjectives
                if token.lemma_ in ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'effective', 'successful']:
                    positive_indicators += 1
                elif token.lemma_ in ['bad', 'terrible', 'awful', 'horrible', 'negative', 'ineffective', 'poor', 'failed']:
                    negative_indicators += 1
        
        if positive_indicators > negative_indicators:
            score = min(0.95, 0.6 + (positive_indicators * 0.1))
            return {'label': 'POSITIVE', 'score': score}
        elif negative_indicators > positive_indicators:
            score = min(0.95, 0.6 + (negative_indicators * 0.1))
            return {'label': 'NEGATIVE', 'score': score}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def simple_sentiment_analysis(self, text: str) -> Dict:
        """Fallback sentiment analysis"""
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
    
    def classify_question(self, question: str) -> str:
        """Enhanced question classification using NLP"""
        if not st.session_state.nlp:
            return self.simple_classify_question(question)
        
        doc = st.session_state.nlp(question.lower())
        
        # Look for question words and patterns
        question_words = [token.lemma_ for token in doc if token.pos_ == "ADV" or token.pos_ == "PRON"]
        
        if any(word in question_words for word in ['what', 'who', 'where', 'when', 'which']):
            return "Factual"
        elif any(word in question_words for word in ['why', 'how']):
            return "Analytical"
        elif any(word in question.lower() for word in ['summarize', 'summary', 'overview', 'main']):
            return "Summary"
        elif any(word in question.lower() for word in ['compare', 'contrast', 'difference', 'similar']):
            return "Comparative"
        else:
            return "General"
    
    def simple_classify_question(self, question: str) -> str:
        """Fallback question classification"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'explain', 'describe', 'who', 'where', 'when']):
            return "Factual"
        elif any(word in question_lower for word in ['why', 'how', 'analyze', 'evaluate']):
            return "Analytical"
        elif any(word in question_lower for word in ['summarize', 'summary', 'overview', 'main']):
            return "Summary"
        else:
            return "General"
    
    def process_document(self, uploaded_file, doc_name: str):
        """Process uploaded document with advanced NLP"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract text
        status_text.text("📄 Extracting text from PDF...")
        progress_bar.progress(15)
        
        text = self.extract_text_from_pdf(uploaded_file)
        if not text:
            st.error("Failed to extract text from PDF")
            return
        
        # Advanced chunking
        status_text.text("🧠 Creating intelligent chunks with NLP...")
        progress_bar.progress(35)
        
        chunks = self.advanced_chunk_text(text, doc_name)
        
        if not chunks:
            st.error("No valid chunks created from document")
            return
        
        # Generate embeddings
        status_text.text("🔮 Generating semantic embeddings...")
        progress_bar.progress(65)
        
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = st.session_state.embedder.encode(chunk_texts, show_progress_bar=False)
        
        # Store in session state
        status_text.text("💾 Storing in vector database...")
        progress_bar.progress(85)
        
        # Add to existing data
        start_idx = len(st.session_state.chunks_data)
        for i, chunk in enumerate(chunks):
            chunk['embedding_idx'] = start_idx + i
            st.session_state.chunks_data.append(chunk)
        
        # Update embeddings matrix
        if st.session_state.embeddings_matrix is None:
            st.session_state.embeddings_matrix = embeddings
        else:
            st.session_state.embeddings_matrix = np.vstack([st.session_state.embeddings_matrix, embeddings])
        
        progress_bar.progress(100)
        status_text.text("✅ Document processed successfully!")
        
        # Store document info with advanced stats
        total_tokens = sum(chunk['token_count'] for chunk in chunks)
        avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0
        
        st.session_state.documents.append({
            'name': doc_name,
            'chunks': len(chunks),
            'text_length': len(text),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': avg_tokens_per_chunk,
            'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    def retrieve_relevant_chunks(self, question: str, top_k: int = 3) -> List[Dict]:
        """Enhanced retrieval with better scoring"""
        if not st.session_state.chunks_data or st.session_state.embeddings_matrix is None:
            return []
        
        # Generate question embedding
        question_embedding = st.session_state.embedder.encode([question])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(question_embedding, st.session_state.embeddings_matrix)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates
        
        # Filter and score chunks
        relevant_chunks = []
        for idx in top_indices[:top_k]:
            if similarities[idx] > 0.15:  # Higher threshold for better quality
                chunk = st.session_state.chunks_data[idx].copy()
                chunk['similarity'] = similarities[idx]
                chunk['distance'] = 1 - similarities[idx]
                chunk['relevance_score'] = self._calculate_relevance_score(question, chunk, similarities[idx])
                relevant_chunks.append(chunk)
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_chunks[:top_k]
    
    def _calculate_relevance_score(self, question: str, chunk: Dict, similarity: float) -> float:
        """Calculate enhanced relevance score"""
        base_score = similarity
        
        # Boost score based on token count (prefer chunks with good content)
        token_bonus = min(0.1, chunk['token_count'] / 200)
        
        # Boost score if chunk has entities that might be relevant
        entity_bonus = 0.0
        if chunk.get('entities'):
            entity_bonus = min(0.1, len(chunk['entities']) * 0.02)
        
        return base_score + token_bonus + entity_bonus
    
    def generate_enhanced_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate better answers using advanced processing"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or upload a relevant document."
        
        question_type = self.classify_question(question)
        
        # Combine contexts more intelligently
        contexts = []
        for chunk in relevant_chunks:
            contexts.append(f"[Page {chunk['page']}] {chunk['text']}")
        
        combined_context = "\n\n".join(contexts)
        
        if question_type == "Summary":
            answer = self._generate_summary_answer(combined_context)
        elif question_type == "Factual":
            answer = self._generate_factual_answer(question, combined_context, relevant_chunks)
        elif question_type == "Analytical":
            answer = self._generate_analytical_answer(question, combined_context)
        else:
            answer = self._generate_general_answer(question, combined_context)
        
        return answer
    
    def _generate_summary_answer(self, context: str) -> str:
        """Generate summary-type answers"""
        sentences = self._get_sentences(context)
        if len(sentences) <= 3:
            return context
        
        # Extract key sentences (simple extractive summarization)
        key_sentences = sentences[:3]  # First 3 sentences
        if len(sentences) > 6:
            key_sentences.extend(sentences[-2:])  # Last 2 sentences
        
        summary = " ".join(key_sentences)
        return f"Based on the document analysis:\n\n{summary}"
    
    def _generate_factual_answer(self, question: str, context: str, chunks: List[Dict]) -> str:
        """Generate factual answers with entity extraction"""
        # Extract key terms from question
        question_terms = set(question.lower().split())
        question_terms = {term for term in question_terms if len(term) > 3}
        
        # Find sentences that contain question terms
        sentences = self._get_sentences(context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())
            if question_terms.intersection(sentence_terms):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Use most relevant sentences
            answer = ". ".join(relevant_sentences[:3]) + "."
        else:
            # Fallback to first chunk
            answer = chunks[0]['text'][:400] + "..." if len(chunks[0]['text']) > 400 else chunks[0]['text']
        
        return f"Based on the document: {answer}"
    
    def _generate_analytical_answer(self, question: str, context: str) -> str:
        """Generate analytical answers"""
        return f"According to the analysis of the document:\n\n{context[:600]}{'...' if len(context) > 600 else ''}\n\nThis information provides insight into your analytical question about the document's content."
    
    def _generate_general_answer(self, question: str, context: str) -> str:
        """Generate general answers"""
        return f"Based on the available information:\n\n{context[:500]}{'...' if len(context) > 500 else ''}"
    
    def ask_question(self, question: str) -> Dict:
        """Enhanced Q&A function"""
        start_time = time.time()
        
        # Enhanced question classification
        question_type = self.classify_question(question)
        
        # Enhanced retrieval
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=3)
        
        # Enhanced answer generation
        answer = self.generate_enhanced_answer(question, relevant_chunks)
        
        # Enhanced confidence calculation
        confidence = self._calculate_answer_confidence(question, relevant_chunks, answer)
        
        # Process time
        process_time = time.time() - start_time
        
        # Enhanced analysis
        sentiment = self.enhanced_sentiment_analysis(answer)
        entities = self._extract_answer_entities(answer)
        
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
    
    def _calculate_answer_confidence(self, question: str, chunks: List[Dict], answer: str) -> float:
        """Calculate enhanced confidence score"""
        if not chunks:
            return 0.2
        
        # Base confidence from similarity scores
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
        base_confidence = avg_similarity
        
        # Boost confidence based on answer length and content
        if len(answer) > 100:
            base_confidence += 0.1
        
        # Boost if answer contains specific information
        if any(chunk.get('entities') for chunk in chunks):
            base_confidence += 0.1
        
        return min(0.95, max(0.2, base_confidence))
    
    def _extract_answer_entities(self, answer: str) -> Dict:
        """Extract entities from the generated answer"""
        if st.session_state.nlp:
            doc = st.session_state.nlp(answer)
            entities = {}
            
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
            
            return entities
        else:
            # Fallback to simple regex-based extraction
            return self._simple_entity_extraction(answer)
    
    def _simple_entity_extraction(self, text: str) -> Dict:
        """Fallback entity extraction"""
        entities = {}
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            entities['DATE'] = list(set(years))
        
        # Extract percentages
        percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
        if percentages:
            entities['PERCENT'] = list(set(percentages))
        
        return entities

# Initialize the system
@st.cache_resource
def get_rag_system():
    return AdvancedRAGSystem()

# Main app
def main():
    st.title("🧠 Advanced RAG Q&A System")
    st.markdown("### Professional-grade document analysis with advanced NLP processing")
    
    # Initialize system
    rag_system = get_rag_system()
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🚀 Loading advanced AI models... (This may take 1-2 minutes on first run)"):
            embedder, nlp, success = AdvancedRAGSystem.load_models()
            if success:
                st.session_state.embedder = embedder
                st.session_state.nlp = nlp
                st.session_state.models_loaded = True
                st.success("✅ Advanced AI models loaded successfully!")
            else:
                st.error("❌ Failed to load some models. Basic functionality will be available.")
                st.session_state.models_loaded = True
    
    # Sidebar
    st.sidebar.header("📁 Document Management")
    
    # Model status
    st.sidebar.subheader("🤖 AI Model Status")
    if st.session_state.embedder:
        st.sidebar.success("✅ Embeddings: all-mpnet-base-v2")
    if st.session_state.nlp and SPACY_AVAILABLE:
        st.sidebar.success("✅ NLP: SpaCy en_core_web_sm")
    elif SPACY_AVAILABLE:
        st.sidebar.warning("⚠️ SpaCy available but model not loaded")
    else:
        st.sidebar.info("ℹ️ SpaCy not available - using fallback")
    
    if NLTK_AVAILABLE:
        st.sidebar.success("✅ NLTK: Available")
    else:
        st.sidebar.info("ℹ️ NLTK not available - using fallback")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document for advanced AI analysis"
    )
    
    if uploaded_file is not None:
        doc_name = uploaded_file.name
        
        if st.sidebar.button("🔄 Process Document with Advanced NLP"):
            if st.session_state.embedder:
                with st.spinner("Processing document with advanced NLP..."):
                    rag_system.process_document(uploaded_file, doc_name)
                    st.sidebar.success(f"✅ Processed: {doc_name}")
                    st.rerun()
            else:
                st.sidebar.error("Please wait for models to load first!")
    
    # Show processed documents with advanced stats
    if st.session_state.documents:
        st.sidebar.subheader("📚 Processed Documents")
        for doc in st.session_state.documents:
            st.sidebar.write(f"📄 **{doc['name']}**")
            st.sidebar.write(f"   - {doc['chunks']} chunks")
            st.sidebar.write(f"   - {doc.get('total_tokens', 'N/A')} tokens")
            st.sidebar.write(f"   - {doc.get('avg_tokens_per_chunk', 0):.1f} avg tokens/chunk")
            st.sidebar.write(f"   - {doc['processed_at']}")
            st.sidebar.write("---")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Advanced Q&A Interface")
        
        # Sample questions
        if st.session_state.documents:
            st.subheader("🎯 Try these advanced questions:")
            sample_questions = [
                "What is the main topic and key findings of this document?",
                "Summarize the methodology and approach used",
                "What are the specific numbers, dates, and statistics mentioned?",
                "Who are the key people and organizations discussed?",
                "What are the main conclusions and recommendations?",
                "How does this document compare different approaches or methods?",
                "What are the limitations or challenges mentioned?"
            ]
            
            selected_question = st.selectbox("Select a sample question:", 
                                           [""] + sample_questions)
        
        # Question input
        question = st.text_input("Or type your own question:", 
                               value=selected_question if 'selected_question' in locals() else "")
        
        # Ask button
        if st.button("🔍 Ask Advanced Question", disabled=not question or not st.session_state.documents):
            if not st.session_state.documents:
                st.error("Please upload and process a document first!")
            elif not st.session_state.embedder:
                st.error("Please wait for AI models to load first!")
            else:
                with st.spinner("🧠 Advanced AI analysis in progress..."):
                    result = rag_system.ask_question(question)
                    
                    # Display answer
                    st.subheader("📝 AI-Generated Answer")
                    st.write(result['answer'])
                    
                    # Display enhanced metadata
                    col_conf, col_type, col_time = st.columns(3)
                    with col_conf:
                        confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.4 else "red"
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col_type:
                        st.metric("Question Type", result['question_type'])
                    with col_time:
                        st.metric("Process Time", f"{result['process_time']:.2f}s")
                    
                    # Display sources with enhanced information
                    if result['sources']:
                        st.subheader("📚 Source Analysis")
                        for i, source in enumerate(result['sources']):
                            relevance = source['similarity']
                            relevance_color = "🟢" if relevance > 0.6 else "🟡" if relevance > 0.4 else "🔴"
                            
                            with st.expander(f"📄 Source {i+1} - Page {source['page']} {relevance_color} Relevance: {relevance:.1%}"):
                                st.write("**Content:**")
                                st.write(source['text'])
                                
                                col_stats1, col_stats2 = st.columns(2)
                                with col_stats1:
                                    st.write(f"**Tokens:** {source.get('token_count', 'N/A')}")
                                    st.write(f"**Sentences:** {source.get('sentence_count', 'N/A')}")
                                with col_stats2:
                                    st.write(f"**Chunk ID:** {source.get('chunk_id', 'N/A')}")
                                    st.write(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                                
                                # Show entities if available
                                if source.get('entities'):
                                    st.write("**🏷️ Entities in this source:**")
                                    for entity_type, entity_list in source['entities'].items():
                                        if entity_list:
                                            st.write(f"- **{entity_type}**: {', '.join(entity_list)}")
                    
                    # Display enhanced NLP analysis
                    if result['entities'] or result['sentiment']:
                        st.subheader("🔍 Advanced NLP Analysis")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            if result['entities']:
                                st.write("**🏷️ Named Entities Extracted:**")
                                for entity_type, entities in result['entities'].items():
                                    if entities:
                                        # Map SpaCy labels to readable names
                                        readable_type = {
                                            'PERSON': 'People',
                                            'ORG': 'Organizations', 
                                            'GPE': 'Places',
                                            'DATE': 'Dates',
                                            'MONEY': 'Money',
                                            'PERCENT': 'Percentages',
                                            'CARDINAL': 'Numbers',
                                            'ORDINAL': 'Ordinals'
                                        }.get(entity_type, entity_type.replace('_', ' ').title())
                                        
                                        st.write(f"- **{readable_type}**: {', '.join(entities)}")
                        
                        with analysis_col2:
                            if result['sentiment']:
                                st.write("**😊 Sentiment Analysis:**")
                                sentiment_emoji = "😊" if result['sentiment']['label'] == 'POSITIVE' else "😞" if result['sentiment']['label'] == 'NEGATIVE' else "😐"
                                sentiment_color = "green" if result['sentiment']['label'] == 'POSITIVE' else "red" if result['sentiment']['label'] == 'NEGATIVE' else "gray"
                                st.write(f"- **Tone**: {sentiment_emoji} :{sentiment_color}[{result['sentiment']['label']}]")
                                st.write(f"- **Confidence**: {result['sentiment']['score']:.3f}")
    
    with col2:
        st.header("📊 Advanced Analytics")
        
        # Document stats with advanced metrics
        if st.session_state.documents:
            st.subheader("📈 Document Intelligence")
            
            # Create enhanced dataframe
            df = pd.DataFrame(st.session_state.documents)
            
            # Enhanced visualizations
            if len(df) > 0:
                # Tokens per document
                if 'total_tokens' in df.columns:
                    fig_tokens = px.bar(df, x='name', y='total_tokens', 
                                      title='Tokens per Document',
                                      labels={'name': 'Document', 'total_tokens': 'Total Tokens'})
                    fig_tokens.update_layout(height=300)
                    st.plotly_chart(fig_tokens, use_container_width=True)
                
                # Average tokens per chunk
                if 'avg_tokens_per_chunk' in df.columns:
                    fig_avg_tokens = px.bar(df, x='name', y='avg_tokens_per_chunk',
                                          title='Average Tokens per Chunk',
                                          labels={'name': 'Document', 'avg_tokens_per_chunk': 'Avg Tokens/Chunk'})
                    fig_avg_tokens.update_layout(height=300)
                    st.plotly_chart(fig_avg_tokens, use_container_width=True)
                
                # Chunks distribution
                fig_chunks = px.bar(df, x='name', y='chunks', 
                                  title='Chunks per Document',
                                  labels={'name': 'Document', 'chunks': 'Number of Chunks'})
                fig_chunks.update_layout(height=300)
                st.plotly_chart(fig_chunks, use_container_width=True)
        
        # Enhanced Q&A History
        if st.session_state.qa_history:
            st.subheader("🕒 Q&A Intelligence")
            
            # Recent questions with enhanced info
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
                confidence_emoji = "🎯" if qa['confidence'] > 0.7 else "⚠️" if qa['confidence'] > 0.4 else "❌"
                with st.expander(f"{confidence_emoji} Q{len(st.session_state.qa_history)-i}: {qa['question'][:35]}..."):
                    st.write(f"**❓ Question:** {qa['question']}")
                    st.write(f"**💡 Answer:** {qa['answer'][:150]}...")
                    st.write(f"**🎯 Confidence:** {qa['confidence']:.1%}")
                    st.write(f"**📊 Type:** {qa['question_type']}")
                    st.write(f"**⏰ Time:** {qa['timestamp']}")
                    st.write(f"**⚡ Speed:** {qa['process_time']:.2f}s")
            
            # Enhanced analytics
            if len(st.session_state.qa_history) > 1:
                # Question type distribution
                question_types = [qa['question_type'] for qa in st.session_state.qa_history]
                type_counts = pd.Series(question_types).value_counts()
                
                fig_types = px.pie(values=type_counts.values, names=type_counts.index,
                                 title='Question Types Distribution')
                fig_types.update_layout(height=300)
                st.plotly_chart(fig_types, use_container_width=True)
                
                # Confidence over time
                if len(st.session_state.qa_history) > 3:
                    confidence_data = pd.DataFrame({
                        'Question': range(1, len(st.session_state.qa_history) + 1),
                        'Confidence': [qa['confidence'] for qa in st.session_state.qa_history]
                    })
                    
                    fig_confidence = px.line(confidence_data, x='Question', y='Confidence',
                                           title='Answer Confidence Trend',
                                           range_y=[0, 1])
                    fig_confidence.update_layout(height=300)
                    st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Enhanced performance metrics
        if st.session_state.qa_history:
            st.subheader("⚡ System Performance")
            
            avg_confidence = np.mean([qa['confidence'] for qa in st.session_state.qa_history])
            avg_time = np.mean([qa['process_time'] for qa in st.session_state.qa_history])
            total_questions = len(st.session_state.qa_history)
            high_confidence_rate = len([qa for qa in st.session_state.qa_history if qa['confidence'] > 0.7]) / total_questions
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}", 
                         delta=f"+{(avg_confidence-0.5)*100:.1f}%" if avg_confidence > 0.5 else None)
                st.metric("Total Questions", total_questions)
            
            with col_metric2:
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
                st.metric("High Confidence Rate", f"{high_confidence_rate:.1%}")
        
        # System information
        st.subheader("🔧 System Info")
        if st.session_state.chunks_data:
            total_chunks = len(st.session_state.chunks_data)
            avg_chunk_tokens = np.mean([chunk.get('token_count', 0) for chunk in st.session_state.chunks_data])
            st.write(f"**Total Chunks:** {total_chunks}")
            st.write(f"**Avg Tokens/Chunk:** {avg_chunk_tokens:.1f}")
        
        if st.session_state.embeddings_matrix is not None:
            st.write(f"**Embedding Dimensions:** {st.session_state.embeddings_matrix.shape[1]}")
            st.write(f"**Vector Database Size:** {st.session_state.embeddings_matrix.shape[0]} vectors")
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("### 🚀 **Advanced AI Technology Stack**")
    
    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
    
    with col_tech1:
        st.markdown("**🧠 NLP Models:**")
        st.markdown("- Sentence-BERT MPNet")
        st.markdown("- SpaCy NLP Pipeline")
        st.markdown("- NLTK Tokenization")
    
    with col_tech2:
        st.markdown("**⚡ Processing:**")
        st.markdown("- Semantic Chunking")
        st.markdown("- Entity Recognition")
        st.markdown("- Advanced Retrieval")
    
    with col_tech3:
        st.markdown("**🎯 Features:**")
        st.markdown("- Multi-type Q&A")
        st.markdown("- Confidence Scoring")
        st.markdown("- Source Attribution")
    
    with col_tech4:
        st.markdown("**📊 Analytics:**")
        st.markdown("- Performance Tracking")
        st.markdown("- Quality Metrics")
        st.markdown("- Usage Analytics")

if __name__ == "__main__":
    main()