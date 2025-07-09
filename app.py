import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
import time
from typing import List, Dict
import pandas as pd
import plotly.express as px
import numpy as np

# Configure page
st.set_page_config(
    page_title="Smart RAG Q&A System",
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
if 'chunks_data' not in st.session_state:
    st.session_state.chunks_data = []
if 'embeddings_matrix' not in st.session_state:
    st.session_state.embeddings_matrix = None

class SmartRAGSystem:
    def __init__(self):
        self.target_chunk_tokens = 100  # Target tokens per chunk
        self.max_chunk_tokens = 200     # Maximum chunk size
        self.overlap_sentences = 2      # Sentences to overlap
        self.min_chunk_tokens = 30      # Minimum viable chunk
    
    @st.cache_resource
    def load_embedder():
        """Load embedding model with caching"""
        return SentenceTransformer('all-mpnet-base-v2')
    
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
    
    def smart_sentence_split(self, text: str) -> List[str]:
        """Advanced sentence splitting using regex patterns"""
        # Handle common abbreviations that shouldn't trigger sentence breaks
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'vs.', 'etc.', 'i.e.', 'e.g.']
        
        # Temporarily replace abbreviations
        temp_text = text
        for i, abbr in enumerate(abbreviations):
            temp_text = temp_text.replace(abbr, f"ABBREV{i}")
        
        # Split on sentence endings followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', temp_text)
        
        # Restore abbreviations
        for i, abbr in enumerate(abbreviations):
            sentences = [s.replace(f"ABBREV{i}", abbr) for s in sentences]
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def count_tokens_simple(self, text: str) -> int:
        """Simple but effective token counting"""
        # Split on whitespace and punctuation, but keep meaningful tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        return len(tokens)
    
    def smart_chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """Smart text chunking with better boundaries"""
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
        """Chunk text from a single page using smart sentence boundaries"""
        chunks = []
        
        # Split into sentences
        sentences = self.smart_sentence_split(text)
        
        if not sentences:
            return chunks
        
        # Group sentences into chunks
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens_simple(sentence)
            
            # Check if adding this sentence exceeds our target
            if (current_chunk_tokens + sentence_tokens > self.max_chunk_tokens and 
                current_chunk_sentences and 
                current_chunk_tokens >= self.min_chunk_tokens):
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(self._create_chunk(chunk_text, page_num, doc_name, len(chunks)))
                
                # Start new chunk with overlap
                if len(current_chunk_sentences) > self.overlap_sentences:
                    overlap_sentences = current_chunk_sentences[-self.overlap_sentences:]
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_chunk_tokens = sum(self.count_tokens_simple(s) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = [sentence]
                    current_chunk_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences and current_chunk_tokens >= self.min_chunk_tokens:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(self._create_chunk(chunk_text, page_num, doc_name, len(chunks)))
        
        return chunks
    
    def _create_chunk(self, text: str, page_num: int, doc_name: str, chunk_id: int) -> Dict:
        """Create a chunk object with metadata"""
        token_count = self.count_tokens_simple(text)
        sentence_count = len(self.smart_sentence_split(text))
        
        # Extract simple entities
        entities = self._extract_simple_entities(text)
        
        return {
            'text': text,
            'page': page_num,
            'document': doc_name,
            'chunk_id': chunk_id,
            'token_count': token_count,
            'char_count': len(text),
            'sentence_count': sentence_count,
            'entities': entities
        }
    
    def _extract_simple_entities(self, text: str) -> Dict:
        """Extract entities using regex patterns"""
        entities = {}
        
        # Extract years (4 digits starting with 19 or 20)
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            entities['DATES'] = list(set([y[0] + y[1] for y in years]))
        
        # Extract percentages
        percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
        if percentages:
            entities['PERCENTAGES'] = list(set(percentages))
        
        # Extract monetary amounts
        money = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*\s*(?:dollars?|euros?|pounds?)\b', text, re.IGNORECASE)
        if money:
            entities['MONEY'] = list(set(money))
        
        # Extract capitalized terms (potential names/organizations)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        # Filter out common words
        stop_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'From', 'According', 'Based'}
        capitalized = [term for term in capitalized if term not in stop_words and len(term) > 2]
        if capitalized:
            entities['NAMES'] = list(set(capitalized[:10]))  # Limit to 10
        
        return entities
    
    def enhanced_sentiment_analysis(self, text: str) -> Dict:
        """Enhanced sentiment analysis using word patterns"""
        # Expanded sentiment word lists
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 
            'success', 'effective', 'beneficial', 'improved', 'better', 'best', 'optimal', 
            'superior', 'outstanding', 'remarkable', 'impressive', 'valuable', 'useful',
            'helpful', 'significant', 'important', 'strong', 'powerful', 'efficient',
            'successful', 'achievement', 'progress', 'advance', 'innovation', 'solution'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'ineffective', 
            'harmful', 'poor', 'disappointing', 'worse', 'worst', 'inferior', 'problematic', 
            'issue', 'problem', 'difficulty', 'challenge', 'concern', 'risk', 'threat',
            'weakness', 'limitation', 'disadvantage', 'decline', 'decrease', 'loss',
            'error', 'mistake', 'fail', 'unsuccessful', 'inadequate', 'insufficient'
        ]
        
        text_lower = text.lower()
        text_words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in positive_words if word in text_words)
        negative_count = sum(1 for word in negative_words if word in text_words)
        
        total_words = len(text_words)
        positive_ratio = positive_count / max(total_words, 1)
        negative_ratio = negative_count / max(total_words, 1)
        
        if positive_count > negative_count and positive_ratio > 0.02:
            score = min(0.95, 0.6 + (positive_ratio * 10))
            return {'label': 'POSITIVE', 'score': score}
        elif negative_count > positive_count and negative_ratio > 0.02:
            score = min(0.95, 0.6 + (negative_ratio * 10))
            return {'label': 'NEGATIVE', 'score': score}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def classify_question(self, question: str) -> str:
        """Enhanced question classification"""
        question_lower = question.lower()
        
        # Factual questions
        if any(word in question_lower for word in ['what', 'who', 'where', 'when', 'which', 'define', 'explain']):
            return "Factual"
        
        # Summary questions
        elif any(word in question_lower for word in ['summarize', 'summary', 'overview', 'main', 'key points', 'gist']):
            return "Summary"
        
        # Analytical questions
        elif any(word in question_lower for word in ['why', 'how', 'analyze', 'compare', 'evaluate', 'assess', 'relationship']):
            return "Analytical"
        
        # Quantitative questions
        elif any(word in question_lower for word in ['how many', 'how much', 'statistics', 'numbers', 'percentage', 'count']):
            return "Quantitative"
        
        else:
            return "General"
    
    def process_document(self, uploaded_file, doc_name: str):
        """Process uploaded document with smart chunking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract text
        status_text.text("📄 Extracting text from PDF...")
        progress_bar.progress(15)
        
        text = self.extract_text_from_pdf(uploaded_file)
        if not text:
            st.error("Failed to extract text from PDF")
            return
        
        # Smart chunking
        status_text.text("🧠 Creating intelligent chunks...")
        progress_bar.progress(35)
        
        chunks = self.smart_chunk_text(text, doc_name)
        
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
        
        # Store document info
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
        
        # Get top candidates (more than needed)
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        
        # Filter and score chunks
        relevant_chunks = []
        for idx in top_indices[:top_k]:
            if similarities[idx] > 0.15:  # Minimum relevance threshold
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
        
        # Boost for optimal token count
        token_count = chunk['token_count']
        if 50 <= token_count <= 150:
            token_bonus = 0.1
        elif 30 <= token_count <= 200:
            token_bonus = 0.05
        else:
            token_bonus = 0
        
        # Boost for entity richness
        entity_bonus = min(0.1, len(chunk.get('entities', {})) * 0.02)
        
        # Boost for question-chunk keyword overlap
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        chunk_words = set(re.findall(r'\b\w+\b', chunk['text'].lower()))
        overlap = len(question_words.intersection(chunk_words))
        overlap_bonus = min(0.1, overlap * 0.01)
        
        return base_score + token_bonus + entity_bonus + overlap_bonus
    
    def generate_smart_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Generate intelligent answers based on question type"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or upload a relevant document."
        
        question_type = self.classify_question(question)
        
        # Prepare context
        contexts = []
        for i, chunk in enumerate(relevant_chunks):
            contexts.append(f"[Source {i+1}, Page {chunk['page']}] {chunk['text']}")
        
        combined_context = "\n\n".join(contexts)
        
        # Generate answer based on question type
        if question_type == "Summary":
            return self._generate_summary_answer(combined_context)
        elif question_type == "Factual":
            return self._generate_factual_answer(question, combined_context, relevant_chunks)
        elif question_type == "Analytical":
            return self._generate_analytical_answer(question, combined_context)
        elif question_type == "Quantitative":
            return self._generate_quantitative_answer(question, combined_context)
        else:
            return self._generate_general_answer(combined_context)
    
    def _generate_summary_answer(self, context: str) -> str:
        """Generate summary answers"""
        sentences = self.smart_sentence_split(context)
        
        if len(sentences) <= 3:
            return f"Summary based on the document:\n\n{context}"
        
        # Extract key sentences (first few and last few)
        key_sentences = sentences[:3]
        if len(sentences) > 6:
            key_sentences.extend(sentences[-2:])
        
        summary = " ".join(key_sentences)
        return f"Summary based on the document:\n\n{summary}"
    
    def _generate_factual_answer(self, question: str, context: str, chunks: List[Dict]) -> str:
        """Generate factual answers"""
        # Extract key terms from question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {word for word in question_words if len(word) > 3}
        
        # Find sentences that best match the question
        sentences = self.smart_sentence_split(context)
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            # Sort by overlap and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [s[0] for s in scored_sentences[:3]]
            answer = ". ".join(best_sentences)
        else:
            # Fallback to first chunk
            answer = chunks[0]['text'][:400] + "..." if len(chunks[0]['text']) > 400 else chunks[0]['text']
        
        return f"Based on the document: {answer}"
    
    def _generate_analytical_answer(self, question: str, context: str) -> str:
        """Generate analytical answers"""
        return f"Analysis based on the document:\n\n{context[:600]}{'...' if len(context) > 600 else ''}\n\nThis information provides insight into your analytical question."
    
    def _generate_quantitative_answer(self, question: str, context: str) -> str:
        """Generate answers focused on numbers and statistics"""
        # Extract numbers, percentages, and quantitative information
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|\s*(?:percent|million|billion|thousand|dollars?|euros?))\b', context, re.IGNORECASE)
        
        if numbers:
            return f"Based on the quantitative information in the document:\n\n{context[:500]}{'...' if len(context) > 500 else ''}\n\nKey numbers found: {', '.join(numbers[:5])}"
        else:
            return f"Based on the document:\n\n{context[:500]}{'...' if len(context) > 500 else ''}"
    
    def _generate_general_answer(self, context: str) -> str:
        """Generate general answers"""
        return f"Based on the available information:\n\n{context[:500]}{'...' if len(context) > 500 else ''}"
    
    def ask_question(self, question: str) -> Dict:
        """Main Q&A function with enhanced processing"""
        start_time = time.time()
        
        # Enhanced question classification
        question_type = self.classify_question(question)
        
        # Enhanced retrieval
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=3)
        
        # Enhanced answer generation
        answer = self.generate_smart_answer(question, relevant_chunks)
        
        # Enhanced confidence calculation
        confidence = self._calculate_answer_confidence(question, relevant_chunks)
        
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
    
    def _calculate_answer_confidence(self, question: str, chunks: List[Dict]) -> float:
        """Calculate comprehensive confidence score"""
        if not chunks:
            return 0.2
        
        # Base confidence from similarity scores
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
        base_confidence = avg_similarity
        
        # Boost for multiple good sources
        if len(chunks) >= 3 and all(chunk.get('similarity', 0) > 0.3 for chunk in chunks):
            base_confidence += 0.1
        
        # Boost for entity-rich sources
        entity_richness = sum(len(chunk.get('entities', {})) for chunk in chunks)
        if entity_richness > 3:
            base_confidence += 0.1
        
        return min(0.95, max(0.2, base_confidence))
    
    def _extract_answer_entities(self, answer: str) -> Dict:
        """Extract entities from the generated answer"""
        return self._extract_simple_entities(answer)

# Initialize the system
@st.cache_resource
def get_rag_system():
    return SmartRAGSystem()

# Main app
def main():
    st.title("🧠 Smart RAG Q&A System")
    st.markdown("### Advanced document analysis with intelligent chunking and semantic search")
    
    # Initialize system
    rag_system = get_rag_system()
    
    # Load embedding model
    if st.session_state.embedder is None:
        with st.spinner("🚀 Loading AI models... (This may take a minute on first run)"):
            st.session_state.embedder = SmartRAGSystem.load_embedder()
            st.success("✅ AI models loaded successfully!")
    
    # Sidebar
    st.sidebar.header("📁 Document Management")
    
    # Model status
    st.sidebar.subheader("🤖 AI Model Status")
    if st.session_state.embedder:
        st.sidebar.success("✅ Embeddings: all-mpnet-base-v2")
        st.sidebar.info("🧠 Using smart regex-based NLP processing")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document for AI analysis"
    )
    
    if uploaded_file is not None:
        doc_name = uploaded_file.name
        
        if st.sidebar.button("🔄 Process Document"):
            if st.session_state.embedder:
                with st.spinner("Processing document with smart chunking..."):
                    rag_system.process_document(uploaded_file, doc_name)
                    st.sidebar.success(f"✅ Processed: {doc_name}")
                    st.rerun()
            else:
                st.sidebar.error("Please wait for models to load first!")
    
    # Show processed documents
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
        st.header("💬 Intelligent Q&A")
        
        # Sample questions
        if st.session_state.documents:
            st.subheader("🎯 Try these intelligent questions:")
            sample_questions = [
                "What is the main topic and key findings?",
                "Summarize the methodology and conclusions",
                "What specific numbers and statistics are mentioned?",
                "Who are the key people and organizations?",
                "What are the main challenges or limitations discussed?",
                "How does this document compare different approaches?",
                "What recommendations or solutions are proposed?"
            ]
            
            selected_question = st.selectbox("Select a sample question:", 
                                           [""] + sample_questions)
        
        # Question input
        question = st.text_input("Or type your own question:", 
                               value=selected_question if 'selected_question' in locals() else "")
        
        # Ask button
        if st.button("🔍 Ask Smart Question", disabled=not question or not st.session_state.documents):
            if not st.session_state.documents:
                st.error("Please upload and process a document first!")
            elif not st.session_state.embedder:
                st.error("Please wait for AI models to load first!")
            else:
                with st.spinner("🧠 AI analysis in progress..."):
                    result = rag_system.ask_question(question)
                    
                    # Display answer
                    st.subheader("📝 Smart Answer")
                    st.write(result['answer'])
                    
                    # Display metadata
                    col_conf, col_type, col_time = st.columns(3)
                    with col_conf:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col_type:
                        st.metric("Question Type", result['question_type'])
                    with col_time:
                        st.metric("Response Time", f"{result['process_time']:.2f}s")
                    
                    # Display sources
                    if result['sources']:
                        st.subheader("📚 Source Analysis")
                        for i, source in enumerate(result['sources']):
                            relevance = source['similarity']
                            relevance_emoji = "🎯" if relevance > 0.6 else "⚠️" if relevance > 0.4 else "❌"
                            
                            with st.expander(f"{relevance_emoji} Source {i+1} - Page {source['page']} (Relevance: {relevance:.1%})"):
                                st.write("**Content:**")
                                st.write(source['text'])
                                
                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    st.write(f"**Tokens:** {source.get('token_count', 'N/A')}")
                                    st.write(f"**Sentences:** {source.get('sentence_count', 'N/A')}")
                                with col_info2:
                                    st.write(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                                    st.write(f"**Chunk ID:** {source.get('chunk_id', 'N/A')}")
                                
                                # Show entities
                                if source.get('entities'):
                                    st.write("**🏷️ Extracted Information:**")
                                    for entity_type, entity_list in source['entities'].items():
                                        if entity_list:
                                            readable_type = entity_type.replace('_', ' ').title()
                                            st.write(f"- **{readable_type}**: {', '.join(entity_list)}")
                    
                    # Display analysis
                    if result['entities'] or result['sentiment']:
                        st.subheader("🔍 AI Analysis")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            if result['entities']:
                                st.write("**🏷️ Key Information Extracted:**")
                                for entity_type, entities in result['entities'].items():
                                    if entities:
                                        readable_type = entity_type.replace('_', ' ').title()
                                        st.write(f"- **{readable_type}**: {', '.join(entities)}")
                        
                        with analysis_col2:
                            if result['sentiment']:
                                st.write("**😊 Sentiment Analysis:**")
                                sentiment_emoji = "😊" if result['sentiment']['label'] == 'POSITIVE' else "😞" if result['sentiment']['label'] == 'NEGATIVE' else "😐"
                                sentiment_color = "green" if result['sentiment']['label'] == 'POSITIVE' else "red" if result['sentiment']['label'] == 'NEGATIVE' else "gray"
                                st.write(f"- **Tone**: {sentiment_emoji} :{sentiment_color}[{result['sentiment']['label']}]")
                                st.write(f"- **Confidence**: {result['sentiment']['score']:.3f}")
    
    with col2:
        st.header("📊 Smart Analytics")
        
        # Document intelligence
        if st.session_state.documents:
            st.subheader("📈 Document Intelligence")
            
            df = pd.DataFrame(st.session_state.documents)
            
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
                    fig_avg = px.bar(df, x='name', y='avg_tokens_per_chunk',
                                   title='Avg Tokens per Chunk',
                                   labels={'name': 'Document', 'avg_tokens_per_chunk': 'Avg Tokens'})
                    fig_avg.update_layout(height=300)
                    st.plotly_chart(fig_avg, use_container_width=True)
        
        # Q&A Intelligence
        if st.session_state.qa_history:
            st.subheader("🧠 Q&A Intelligence")
            
            # Recent questions
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
                confidence_emoji = "🎯" if qa['confidence'] > 0.7 else "⚠️" if qa['confidence'] > 0.4 else "❌"
                with st.expander(f"{confidence_emoji} Q{len(st.session_state.qa_history)-i}: {qa['question'][:35]}..."):
                    st.write(f"**❓ Question:** {qa['question']}")
                    st.write(f"**💡 Answer:** {qa['answer'][:150]}...")
                    st.write(f"**🎯 Confidence:** {qa['confidence']:.1%}")
                    st.write(f"**📊 Type:** {qa['question_type']}")
                    st.write(f"**⏰ Time:** {qa['timestamp']}")
            
            # Analytics charts
            if len(st.session_state.qa_history) > 1:
                # Question types
                question_types = [qa['question_type'] for qa in st.session_state.qa_history]
                type_counts = pd.Series(question_types).value_counts()
                
                fig_types = px.pie(values=type_counts.values, names=type_counts.index,
                                 title='Question Types')
                fig_types.update_layout(height=300)
                st.plotly_chart(fig_types, use_container_width=True)
                
                # Confidence trend
                if len(st.session_state.qa_history) > 3:
                    confidence_data = pd.DataFrame({
                        'Question': range(1, len(st.session_state.qa_history) + 1),
                        'Confidence': [qa['confidence'] for qa in st.session_state.qa_history]
                    })
                    
                    fig_conf = px.line(confidence_data, x='Question', y='Confidence',
                                     title='Answer Confidence Trend',
                                     range_y=[0, 1])
                    fig_conf.update_layout(height=300)
                    st.plotly_chart(fig_conf, use_container_width=True)
        
        # Performance metrics
        if st.session_state.qa_history:
            st.subheader("⚡ Performance Metrics")
            
            avg_confidence = np.mean([qa['confidence'] for qa in st.session_state.qa_history])
            avg_time = np.mean([qa['process_time'] for qa in st.session_state.qa_history])
            total_questions = len(st.session_state.qa_history)
            high_conf_rate = len([qa for qa in st.session_state.qa_history if qa['confidence'] > 0.7]) / total_questions
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                st.metric("Total Questions", total_questions)
            with col_m2:
                st.metric("Avg Response Time", f"{avg_time:.2f}s") 
                st.metric("High Confidence Rate", f"{high_conf_rate:.1%}")
        
        # System information
        st.subheader("🔧 System Status")
        if st.session_state.chunks_data:
            total_chunks = len(st.session_state.chunks_data)
            avg_chunk_tokens = np.mean([chunk.get('token_count', 0) for chunk in st.session_state.chunks_data])
            st.write(f"**Total Chunks:** {total_chunks}")
            st.write(f"**Avg Tokens/Chunk:** {avg_chunk_tokens:.1f}")
        
        if st.session_state.embeddings_matrix is not None:
            st.write(f"**Embedding Dimensions:** {st.session_state.embeddings_matrix.shape[1]}")
            st.write(f"**Vector Database Size:** {st.session_state.embeddings_matrix.shape[0]}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 **Smart AI Technology Stack**")
    
    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
    
    with col_tech1:
        st.markdown("**🧠 AI Models:**")
        st.markdown("- MPNet Embeddings")
        st.markdown("- Smart Chunking")
        st.markdown("- Regex NLP")
    
    with col_tech2:
        st.markdown("**⚡ Processing:**")
        st.markdown("- Token-Based Sizing")
        st.markdown("- Sentence Boundaries")
        st.markdown("- Intelligent Overlap")
    
    with col_tech3:
        st.markdown("**🎯 Features:**")
        st.markdown("- Multi-Type Q&A")
        st.markdown("- Confidence Scoring")
        st.markdown("- Entity Extraction")
    
    with col_tech4:
        st.markdown("**📊 Analytics:**")
        st.markdown("- Performance Tracking")
        st.markdown("- Quality Metrics")
        st.markdown("- Usage Analytics")

if __name__ == "__main__":
    main()