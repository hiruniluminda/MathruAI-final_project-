import groq
import os
from dotenv import load_dotenv
import re
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
import faiss
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import json
from typing import List, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import tiktoken

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

load_dotenv()

def extract_text_from_pdf(file_path):
    """Extract all text from a PDF file with better error handling"""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
    except Exception as e:
        print(f"Error extracting from PDF: {str(e)}")
    return text

class TokenManager:
    """Manages token counting and context truncation"""
    
    def __init__(self, model_name="llama3-8b-8192", max_context_tokens=3000):
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            print("Warning: tiktoken not available, using approximate token counting")
    
    def count_tokens(self, text):
        """Count tokens in text accurately"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate token count (1 token ≈ 4 characters)
            return len(text) // 4
    
    def truncate_context(self, context_chunks, query, system_prompt):
        """Truncate context to fit within token limits"""
        if not context_chunks:
            return ""
        
        # Reserve tokens for system prompt, query, and response
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        response_buffer = 500  # Reserve for response
        
        available_tokens = self.max_context_tokens - system_tokens - query_tokens - response_buffer
        
        if available_tokens <= 0:
            print("Warning: Query too long, using minimal context")
            return context_chunks[0][:500]  # Use first 500 chars as fallback
        
        # Build context within token limit
        truncated_context = ""
        current_tokens = 0
        
        for chunk in context_chunks:
            chunk_tokens = self.count_tokens(chunk)
            
            if current_tokens + chunk_tokens <= available_tokens:
                if truncated_context:
                    truncated_context += "\n\n"
                truncated_context += chunk
                current_tokens += chunk_tokens
            else:
                # Try to fit partial chunk
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 50:  # Only if we have reasonable space
                    # Estimate characters we can fit
                    chars_to_fit = remaining_tokens * 4  # Approximate
                    partial_chunk = chunk[:chars_to_fit]
                    
                    # Try to end at a sentence boundary
                    sentences = sent_tokenize(partial_chunk)
                    if len(sentences) > 1:
                        partial_chunk = ' '.join(sentences[:-1])
                    
                    if truncated_context:
                        truncated_context += "\n\n"
                    truncated_context += partial_chunk
                break
        
        final_tokens = self.count_tokens(truncated_context)
        print(f"Context truncated to {final_tokens} tokens (limit: {available_tokens})")
        
        return truncated_context

class TextChunker:
    """Advanced text chunking with multiple strategies"""
    
    def __init__(self, max_chunk_size=800, overlap_size=100, min_chunk_size=100):  # Reduced chunk size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text):
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate token count
            return len(text.split()) * 1.3
    
    def chunk_by_sentences(self, text):
        """Chunk text by sentences with overlap"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max size, finalize current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                
                # Add sentences from the end for overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent_len = len(current_chunk[i])
                    if overlap_length + sent_len <= self.overlap_size:
                        overlap_sentences.insert(0, current_chunk[i])
                        overlap_length += sent_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk_by_paragraphs(self, text):
        """Chunk text by paragraphs with size limits"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            # If paragraph alone exceeds max size, split it further
            if para_length > self.max_chunk_size:
                # Finalize current chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph by sentences
                para_chunks = self.chunk_by_sentences(paragraph)
                chunks.extend(para_chunks)
                continue
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_length + para_length > self.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with potential overlap
                if self.overlap_size > 0 and current_chunk:
                    # Keep last paragraph for overlap if it fits
                    last_para = current_chunk[-1]
                    if len(last_para) <= self.overlap_size:
                        current_chunk = [last_para]
                        current_length = len(last_para)
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(paragraph)
            current_length += para_length + 2  # +2 for \n\n
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def smart_chunk(self, text):
        """Intelligent chunking that tries multiple strategies"""
        # Clean the text first
        text = self.clean_text(text)
        
        if len(text) <= self.max_chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        # Try paragraph-based chunking first
        chunks = self.chunk_by_paragraphs(text)
        
        # If chunks are still too large, use sentence-based chunking
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                sentence_chunks = self.chunk_by_sentences(chunk)
                final_chunks.extend(sentence_chunks)
            else:
                final_chunks.append(chunk)
        
        # Filter out chunks that are too small
        return [chunk for chunk in final_chunks if len(chunk) >= self.min_chunk_size]
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)    # Add space after period if missing
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.db_name = os.getenv('MYSQL_DATABASE', 'rag_system')
        self.connect()
        self.setup_tables()

    def connect(self):
        """Connect to MySQL server and create database if it doesn't exist"""
        try:
            temp_connection = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST', 'localhost'),
                user=os.getenv('MYSQL_USER', 'root'),
                password=os.getenv('MYSQL_PASSWORD', '20000624')
            )
            print("Connected to MySQL server")

            cursor = temp_connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.db_name}`")
            print(f"Database `{self.db_name}` ensured")

            cursor.close()
            temp_connection.close()

            self.connection = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST', 'localhost'),
                database=self.db_name,
                user=os.getenv('MYSQL_USER', 'root'),
                password=os.getenv('MYSQL_PASSWORD', '20000624')
            )
            print(f"Connected to MySQL database `{self.db_name}`")

        except Error as e:
            print(f"MySQL connection error: {e}")
            self.connection = None

    def setup_tables(self):
        """Create necessary tables with proper field sizes"""
        if not self.connection:
            print("No connection: Skipping table setup")
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Create document_chunks table with LONGTEXT for large chunks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chunk_text LONGTEXT NOT NULL,
                    source_file VARCHAR(255),
                    chunk_index INT,
                    chunk_size INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding_vector_id INT,
                    metadata JSON,
                    chunk_hash VARCHAR(64),
                    INDEX idx_source_file (source_file),
                    INDEX idx_chunk_index (chunk_index),
                    INDEX idx_embedding_id (embedding_vector_id)
                )
            """)
            
            # Create search_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response LONGTEXT,
                    relevant_chunks_count INT,
                    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    similarity_threshold FLOAT,
                    top_k INT,
                    response_time_ms INT,
                    context_tokens INT,
                    INDEX idx_timestamp (search_timestamp)
                )
            """)
            
            self.connection.commit()
            print("✅ Database tables setup complete")
            
        except Error as e:
            print(f"❌ Error setting up tables: {e}")

    def store_chunk(self, chunk_text: str, source_file: str, chunk_index: int, 
                   embedding_vector_id: int, metadata: dict = None) -> int:
        """Store a document chunk with metadata and size tracking"""
        if not self.connection:
            return -1
        
        try:
            # Calculate chunk hash for deduplication
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            chunk_size = len(chunk_text)
            
            cursor = self.connection.cursor()
            
            # Check if chunk already exists
            cursor.execute("""
                SELECT id FROM document_chunks WHERE chunk_hash = %s
            """, (chunk_hash,))
            
            if cursor.fetchone():
                print(f"Chunk already exists (hash: {chunk_hash[:8]}...)")
                return -1
            
            cursor.execute("""
                INSERT INTO document_chunks 
                (chunk_text, source_file, chunk_index, chunk_size, embedding_vector_id, metadata, chunk_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (chunk_text, source_file, chunk_index, chunk_size, embedding_vector_id, 
                  json.dumps(metadata) if metadata else None, chunk_hash))
            
            self.connection.commit()
            chunk_id = cursor.lastrowid
            print(f"Stored chunk {chunk_id} ({chunk_size} chars)")
            return chunk_id
            
        except Error as e:
            print(f"Error storing chunk: {e}")
            return -1
    
    def log_search(self, query: str, response: str, chunks_count: int, 
                   similarity_threshold: float, top_k: int, response_time_ms: int = None, context_tokens: int = None):
        """Log search query and response with performance metrics"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO search_logs 
                (query, response, relevant_chunks_count, similarity_threshold, top_k, response_time_ms, context_tokens)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (query, response, chunks_count, similarity_threshold, top_k, response_time_ms, context_tokens))
            
            self.connection.commit()
            
        except Error as e:
            print(f"Error logging search: {e}")
    
    def get_chunk_stats(self) -> dict:
        """Get comprehensive chunk statistics"""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(chunk_size), MIN(chunk_size), MAX(chunk_size) FROM document_chunks")
            size_stats = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(DISTINCT source_file) FROM document_chunks")
            unique_sources = cursor.fetchone()[0]
            
            return {
                "total_chunks": total_chunks,
                "unique_sources": unique_sources,
                "avg_chunk_size": int(size_stats[0]) if size_stats[0] else 0,
                "min_chunk_size": size_stats[1] or 0,
                "max_chunk_size": size_stats[2] or 0
            }
            
        except Error as e:
            print(f"Error getting chunk stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

class VectorPregnancyRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', chunk_size=800, chunk_overlap=100):  # Reduced sizes
        # Initialize API client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = groq.Groq(api_key=api_key)
        self.knowledge_base = []
        
        # Initialize token manager
        self.token_manager = TokenManager(max_context_tokens=3000)  # Conservative limit
        
        # Initialize text chunker with smaller sizes
        self.chunker = TextChunker(
            max_chunk_size=chunk_size,
            overlap_size=chunk_overlap,
            min_chunk_size=50
        )
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {embedding_model}, dimension: {self.embedding_dim}")
        
        # Initialize FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
        
        # Initialize database manager
        try:
            self.db_manager = DatabaseManager()
        except Exception as e:
            print(f"Database initialization failed: {e}")
            self.db_manager = None
        
        # File paths for persistence
        self.kb_file = 'data/knowledge_base.pkl'
        self.faiss_index_file = 'data/faiss_index.bin'
        self.hash_file = 'data/kb_hash.txt'
        
        # Load existing knowledge base
        self.load_knowledge_base()
    
    def _calculate_kb_hash(self):
        """Calculate hash of current knowledge base for change detection"""
        kb_string = '\n'.join(self.knowledge_base)
        return hashlib.md5(kb_string.encode()).hexdigest()
    
    def _load_cached_data(self):
        """Load cached knowledge base and FAISS index if available"""
        try:
            if not all(os.path.exists(f) for f in [self.kb_file, self.faiss_index_file, self.hash_file]):
                return False
            
            with open(self.kb_file, 'rb') as f:
                cached_kb = pickle.load(f)
            
            with open(self.hash_file, 'r') as f:
                cached_hash = f.read().strip()
            
            current_hash = hashlib.md5('\n'.join(cached_kb).encode()).hexdigest()
            
            if cached_hash == current_hash:
                self.knowledge_base = cached_kb
                self.faiss_index = faiss.read_index(self.faiss_index_file)
                
                print(f"Loaded cached knowledge base with {len(self.knowledge_base)} chunks")
                print(f"FAISS index loaded with {self.faiss_index.ntotal} vectors")
                return True
            else:
                print("Cache hash mismatch, will regenerate embeddings")
                return False
                
        except Exception as e:
            print(f"Error loading cached data: {e}")
            return False
    
    def _save_cached_data(self):
        """Save knowledge base and FAISS index to cache"""
        try:
            os.makedirs('data', exist_ok=True)
            
            with open(self.kb_file, 'wb') as f:
                pickle.dump(self.knowledge_base, f)
            
            faiss.write_index(self.faiss_index, self.faiss_index_file)
            
            with open(self.hash_file, 'w') as f:
                f.write(self._calculate_kb_hash())
            
            print("Cached knowledge base and FAISS index")
            
        except Exception as e:
            print(f"Error saving cached data: {e}")
    
    def _generate_and_store_embeddings(self, texts: List[str], source_file: str = None):
        """Generate embeddings and store them in FAISS index"""
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index with IDs
        start_id = len(self.knowledge_base)
        ids = np.arange(start_id, start_id + len(texts), dtype=np.int64)
        self.faiss_index.add_with_ids(embeddings, ids)
        
        # Store in database if available
        if self.db_manager:
            for i, (text, embedding_id) in enumerate(zip(texts, ids)):
                self.db_manager.store_chunk(
                    chunk_text=text,
                    source_file=source_file or "default",
                    chunk_index=i,
                    embedding_vector_id=int(embedding_id),
                    metadata={"chunk_method": "smart_chunk", "embedding_model": "all-MiniLM-L6-v2"}
                )
        
        print(f"Added {len(texts)} vectors to FAISS index")
    
    def update_knowledge_base_from_pdf(self, pdf_path):
        """Update the knowledge base with smart chunking from PDF"""
        content = extract_text_from_pdf(pdf_path)
        if not content:
            print("No content found in the PDF.")
            return False
        
        print(f"Extracted {len(content)} characters from PDF")
        
        # Use smart chunking with smaller chunks
        new_chunks = self.chunker.smart_chunk(content)
        
        if not new_chunks:
            print("No valid content chunks found in the PDF.")
            return False
        
        print(f"Created {len(new_chunks)} chunks using smart chunking")
        
        # Show chunk size distribution
        chunk_sizes = [len(chunk) for chunk in new_chunks]
        print(f"Chunk sizes - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)//len(chunk_sizes)}")
        
        before = len(self.knowledge_base)
        self.knowledge_base.extend(new_chunks)
        
        # Generate and store embeddings
        source_filename = os.path.basename(pdf_path)
        self._generate_and_store_embeddings(new_chunks, source_filename)
        
        after = len(self.knowledge_base)
        print(f"Added {after - before} new chunks; total is now {after}.")
        
        # Save updated cache
        self._save_cached_data()
        return True

    def load_knowledge_base(self):
        """Load and process the pregnancy guide knowledge base"""
        if self._load_cached_data():
            return
        
        try:
            with open('data/pregnancy_guide.txt', 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Use smart chunking for initial knowledge base
            self.knowledge_base = self.chunker.smart_chunk(content)
            
        except FileNotFoundError:
            print("Knowledge base file not found. Creating default knowledge base.")
            self.knowledge_base = [
                "General pregnancy information will be provided based on medical guidelines.",
                "Always consult your healthcare provider for medical advice during pregnancy.",
                "Maintain a healthy diet with prenatal vitamins during pregnancy.",
                "Regular prenatal checkups are essential for a healthy pregnancy."
            ]
        
        if self.knowledge_base:
            self._generate_and_store_embeddings(self.knowledge_base, "default_knowledge_base")
            print(f"Initialized FAISS index with {len(self.knowledge_base)} knowledge chunks")
            self._save_cached_data()
        else:
            print("No knowledge base content available")
    
    def find_relevant_context(self, query, top_k=5, similarity_threshold=0.1):  # Increased top_k to get more options
        """Find most relevant context using FAISS vector similarity with token management"""
        if not self.knowledge_base or self.faiss_index.ntotal == 0:
            return ""
        
        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search for more chunks initially
            search_k = min(top_k * 3, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_embedding, search_k)
            
            relevant_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= similarity_threshold:
                    relevant_chunks.append(self.knowledge_base[idx])
                    print(f"Retrieved chunk {idx} with similarity: {score:.3f}")
            
            if not relevant_chunks and len(indices[0]) > 0 and indices[0][0] != -1:
                best_idx = indices[0][0]
                relevant_chunks = [self.knowledge_base[best_idx]]
                print(f"Using best match (chunk {best_idx}) with similarity: {scores[0][0]:.3f}")
            
            # Use token manager to truncate context appropriately
            system_prompt = "You are a helpful pregnancy guidance assistant. Provide accurate, supportive, and safe information about pregnancy."
            truncated_context = self.token_manager.truncate_context(relevant_chunks, query, system_prompt)
            
            return truncated_context
            
        except Exception as e:
            print(f"Error in FAISS retrieval: {e}")
            return self._fallback_keyword_search(query, top_k)
    
    def _fallback_keyword_search(self, query, top_k=3):
        """Fallback keyword matching method with token management"""
        print("Using fallback keyword search")
        query_lower = query.lower()
        scored_chunks = []
        
        for chunk in self.knowledge_base:
            chunk_lower = chunk.lower()
            score = 0
            query_words = re.findall(r'\w+', query_lower)
            
            for word in query_words:
                if len(word) > 3:
                    score += chunk_lower.count(word)
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        relevant_chunks = [chunk for score, chunk in scored_chunks[:top_k * 2]]  # Get more for truncation
        
        # Use token manager to truncate
        system_prompt = "You are a helpful pregnancy guidance assistant."
        return self.token_manager.truncate_context(relevant_chunks, query, system_prompt)
    
    def get_kb_stats(self):
        """Get comprehensive statistics about the knowledge base"""
        base_stats = {
            "total_chunks": len(self.knowledge_base),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "embedding_dimension": self.embedding_dim,
            "embedding_model": "all-MiniLM-L6-v2",
            "database_connected": self.db_manager is not None and self.db_manager.connection is not None,
            "max_context_tokens": self.token_manager.max_context_tokens
        }
        
        # Add database stats if available
        if self.db_manager:
            db_stats = self.db_manager.get_chunk_stats()
            base_stats.update(db_stats)
        
        return base_stats
    
    def generate_response(self, query):
        """Generate response using RAG approach with proper token management"""
        start_time = datetime.now()
        
        try:
            context = self.find_relevant_context(query)
            context_tokens = self.token_manager.count_tokens(context)
            
            system_prompt = """You are a helpful pregnancy guidance assistant. Provide accurate, supportive, and safe information about pregnancy. Always recommend consulting healthcare providers for medical concerns. Be empathetic and understanding."""
            
            user_prompt = f"""Context information:
{context}

Question: {query}

Please provide a helpful response based on the context above. If the context doesn't contain relevant information, provide general pregnancy guidance while emphasizing the importance of consulting healthcare providers."""
            
            # Final token check
            total_tokens = (
                self.token_manager.count_tokens(system_prompt) + 
                self.token_manager.count_tokens(user_prompt) + 
                500  # Reserve for response
            )
            
            print(f"Total estimated tokens: {total_tokens}")
            
            if total_tokens > 5500:  # Leave buffer for Groq's 6000 limit
                print("Warning: Still close to token limit, further reducing context")
                # Emergency context reduction
                context_lines = context.split('\n\n')
                context = '\n\n'.join(context_lines[:2])  # Keep only first 2 chunks
                user_prompt = f"""Context information:
{context}

Question: {query}

Please provide a helpful response based on the context above."""
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log the search if database is available
            if self.db_manager:
                chunks_count = len(context.split('\n\n')) if context else 0
                self.db_manager.log_search(
                    query, response_text, chunks_count, 0.1, 5, 
                    int(response_time), context_tokens
                )
            
            return response_text
            
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}. Please try again or consult your healthcare provider."
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if self.db_manager:
                self.db_manager.log_search(
                    query, error_msg, 0, 0.1, 5, 
                    int(response_time), 0
                )
            
            return error_msg
    
    def generate_response_streaming(self, query):
        """Generate streaming response for better user experience"""
        try:
            context = self.find_relevant_context(query)
            context_tokens = self.token_manager.count_tokens(context)
            
            system_prompt = """You are a helpful pregnancy guidance assistant. Provide accurate, supportive, and safe information about pregnancy. Always recommend consulting healthcare providers for medical concerns. Be empathetic and understanding."""
            
            user_prompt = f"""Context information:
{context}

Question: {query}

Please provide a helpful response based on the context above. If the context doesn't contain relevant information, provide general pregnancy guidance while emphasizing the importance of consulting healthcare providers."""
            
            # Stream the response
            stream = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Log the complete response
            if self.db_manager:
                chunks_count = len(context.split('\n\n')) if context else 0
                self.db_manager.log_search(
                    query, full_response, chunks_count, 0.1, 5, 
                    None, context_tokens
                )
                
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}. Please try again or consult your healthcare provider."
            yield error_msg
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()

# For backward compatibility
PregnancyRAG = VectorPregnancyRAG

# Example usage and testing
# if __name__ == "__main__":
#     # Initialize the RAG system
#     rag = VectorPregnancyRAG(chunk_size=600, chunk_overlap=80)
    
#     # Test with various queries
#     test_queries = [
#         "What are the early signs of pregnancy?",
#         "How much weight should I gain during pregnancy?",
#         "What foods should I avoid while pregnant?",
#         "When should I start taking prenatal vitamins?"
#     ]
    
#     print("Testing RAG system with token management...")
#     print("=" * 50)
    
#     for query in test_queries:
#         print(f"\nQuery: {query}")
#         print("-" * 30)
        
#         try:
#             response = rag.generate_response(query)
#             print(f"Response: {response}")
#         except Exception as e:
#             print(f"Error: {e}")
        
#         print("-" * 30)
    
#     # Print system statistics
#     stats = rag.get_kb_stats()
#     print(f"\nSystem Statistics:")
#     print(f"Total chunks: {stats.get('total_chunks', 0)}")
#     print(f"Max context tokens: {stats.get('max_context_tokens', 0)}")
#     print(f"Database connected: {stats.get('database_connected', False)}")
    
#     # Cleanup
#     del rag