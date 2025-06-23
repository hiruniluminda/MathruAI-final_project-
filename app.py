from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_system import VectorPregnancyRAG
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import json
from typing import Optional, Dict, Any
import traceback

os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

app = Flask(__name__)
CORS(app, origins=['*'])  # Configure CORS more specifically in production

# Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize RAG system with error handling
rag_system: Optional[VectorPregnancyRAG] = None

def initialize_rag_system():
    """Initialize RAG system with comprehensive error handling"""
    global rag_system
    try:
        logger.info("Initializing RAG system...")
        rag_system = VectorPregnancyRAG(
            embedding_model='all-MiniLM-L6-v2',
            chunk_size=1000,
            chunk_overlap=200
        )
        logger.info("RAG system initialized successfully")
        
        # Log initial stats
        stats = rag_system.get_kb_stats()
        logger.info(f"Initial KB stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        rag_system = None

# Initialize on startup
initialize_rag_system()

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_rag_system() -> tuple[bool, Optional[str]]:
    """Validate RAG system availability"""
    if not rag_system:
        return False, "RAG system not initialized"
    return True, None

def create_error_response(message: str, status_code: int = 500) -> tuple:
    """Create standardized error response"""
    return jsonify({
        "error": message,
        "status": "error",
        "timestamp": datetime.now().isoformat()
    }), status_code

def create_success_response(data: Dict[Any, Any], message: str = "Success") -> tuple:
    """Create standardized success response"""
    response_data = {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    response_data.update(data)
    return jsonify(response_data), 200

@app.route('/', methods=['GET'])
def index():
    """Enhanced API documentation and status"""
    system_status = "healthy" if rag_system else "unhealthy"
    
    api_info = {
        "name": "Enhanced Pregnancy RAG API",
        "version": "2.1",
        "status": system_status,
        "description": "Advanced RAG system with FAISS vector search and MySQL storage",
        "features": [
            "FAISS vector indexing for fast similarity search",
            "MySQL database for metadata and search logging",
            "Advanced semantic search with configurable parameters",
            "PDF document processing and ingestion",
            "Smart text chunking with overlap",
            "Query analytics and performance monitoring",
            "Robust caching and persistence",
            "Comprehensive error handling and logging"
        ],
        "endpoints": {
            "/": "GET - API documentation and status",
            "/health": "GET - Comprehensive system health check",
            "/stats": "GET - Knowledge base and system statistics",
            "/chat": "POST - Interactive chat with AI assistant",
            "/search": "POST - Advanced semantic search",
            "/upload": "POST - Upload and process PDF documents",
            "/analytics": "GET - Search analytics and usage metrics",
            "/reinitialize": "POST - Reinitialize RAG system (admin)"
        },
        "chat_parameters": {
            "message": "Required - User message/question",
            "top_k": "Optional - Number of relevant chunks to retrieve (default: 3)",
            "similarity_threshold": "Optional - Minimum similarity score (default: 0.1)"
        },
        "search_parameters": {
            "query": "Required - Search query",
            "top_k": "Optional - Number of results (default: 3)",
            "similarity_threshold": "Optional - Minimum similarity score (default: 0.1)"
        }
    }
    
    if rag_system:
        try:
            stats = rag_system.get_kb_stats()
            api_info["current_stats"] = stats
        except Exception as e:
            api_info["stats_error"] = str(e)
    
    return jsonify(api_info)

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_system_ready": rag_system is not None,
        "components": {}
    }
    
    if rag_system:
        try:
            stats = rag_system.get_kb_stats()
            health_info["components"] = {
                "knowledge_base": {
                    "status": "healthy" if stats["total_chunks"] > 0 else "empty",
                    "total_chunks": stats["total_chunks"]
                },
                "faiss_index": {
                    "status": "healthy" if stats["faiss_index_size"] > 0 else "empty",
                    "vectors": stats["faiss_index_size"],
                    "dimension": stats["embedding_dimension"]
                },
                "database": {
                    "status": "healthy" if stats["database_connected"] else "disconnected",
                    "connected": stats["database_connected"]
                },
                "embedding_model": {
                    "status": "healthy",
                    "model": stats["embedding_model"]
                }
            }
            
            # Overall health assessment
            if (stats["total_chunks"] == 0 or 
                stats["faiss_index_size"] == 0 or 
                not stats["database_connected"]):
                health_info["status"] = "degraded"
                
        except Exception as e:
            health_info["error"] = str(e)
            health_info["status"] = "unhealthy"
            logger.error(f"Health check error: {str(e)}")
    else:
        health_info["status"] = "unhealthy"
        health_info["error"] = "RAG system not initialized"
    
    status_code = 200 if health_info["status"] in ["healthy", "degraded"] else 503
    return jsonify(health_info), status_code

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get comprehensive system statistics"""
    is_valid, error_msg = validate_rag_system()
    if not is_valid:
        return create_error_response(error_msg, 503)
    
    try:
        stats = rag_system.get_kb_stats()
        
        # Enhanced system information
        system_info = {
            "system_status": "operational",
            "initialization_time": datetime.now().isoformat(),
            "features_enabled": [
                "FAISS vector search",
                "MySQL metadata storage", 
                "Semantic similarity search",
                "PDF document processing",
                "Query logging and analytics",
                "Smart text chunking",
                "Embedding caching"
            ],
            "performance_metrics": {
                "embedding_model": stats.get("embedding_model", "all-MiniLM-L6-v2"),
                "embedding_dimension": stats.get("embedding_dimension", 384),
                "index_type": "FAISS IndexFlatIP with IDMap"
            }
        }
        
        return create_success_response({
            "knowledge_base_stats": stats,
            "system_info": system_info
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return create_error_response(f"Failed to retrieve stats: {str(e)}")

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with configurable parameters"""
    is_valid, error_msg = validate_rag_system()
    if not is_valid:
        return create_error_response(error_msg, 503)
        
    try:
        data = request.get_json()
        
        if not data:
            return create_error_response("Request body must be JSON", 400)
            
        if 'message' not in data:
            return create_error_response("'message' field is required", 400)
            
        user_message = data['message'].strip()
        
        if not user_message:
            return create_error_response("Message cannot be empty", 400)
            
        # Extract optional parameters
        top_k = data.get('top_k', 3)
        similarity_threshold = data.get('similarity_threshold', 0.1)
        
        # Validate parameters
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            return create_error_response("top_k must be an integer between 1 and 10", 400)
            
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0 or similarity_threshold > 1:
            return create_error_response("similarity_threshold must be a number between 0 and 1", 400)
        
        logger.info(f"Chat request - Query: {user_message[:100]}..., top_k: {top_k}, threshold: {similarity_threshold}")
        
        start_time = datetime.now()
        
        # Store original method and temporarily modify retrieval parameters
        original_method = rag_system.find_relevant_context
        
        def custom_find_context(query):
            return original_method(query, top_k=top_k, similarity_threshold=similarity_threshold)
        
        # Temporarily replace method
        rag_system.find_relevant_context = custom_find_context
        
        try:
            # Generate response
            response_text = rag_system.generate_response(user_message)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return create_success_response({
                "response": response_text,
                "processing_time_seconds": round(processing_time, 3),
                "search_method": "FAISS vector similarity + MySQL logging",
                "parameters_used": {
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold
                }
            })
            
        finally:
            # Restore original method
            rag_system.find_relevant_context = original_method
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_error_response(f"Chat processing failed: {str(e)}")

@app.route('/search', methods=['POST'])
def search_knowledge():
    """Advanced semantic search with FAISS"""
    is_valid, error_msg = validate_rag_system()
    if not is_valid:
        return create_error_response(error_msg, 503)
        
    try:
        data = request.get_json()
        
        if not data:
            return create_error_response("Request body must be JSON", 400)
            
        if 'query' not in data:
            return create_error_response("'query' field is required", 400)
            
        query = data['query'].strip()
        top_k = data.get('top_k', 3)
        similarity_threshold = data.get('similarity_threshold', 0.1)
        
        if not query:
            return create_error_response("Query cannot be empty", 400)
            
        # Validate parameters
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            return create_error_response("top_k must be an integer between 1 and 20", 400)
            
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0 or similarity_threshold > 1:
            return create_error_response("similarity_threshold must be a number between 0 and 1", 400)
            
        logger.info(f"Search request - Query: {query[:100]}..., top_k: {top_k}, threshold: {similarity_threshold}")
        
        start_time = datetime.now()
        
        # Perform search
        context = rag_system.find_relevant_context(
            query, 
            top_k=top_k, 
            similarity_threshold=similarity_threshold
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Split context into chunks for display
        chunks = [chunk.strip() for chunk in context.split('\n\n') if chunk.strip()]
        
        return create_success_response({
            "query": query,
            "relevant_chunks": chunks,
            "num_chunks_found": len(chunks),
            "processing_time_seconds": round(processing_time, 3),
            "search_method": "FAISS vector similarity",
            "parameters": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
        })
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return create_error_response(f"Search failed: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF files"""
    is_valid, error_msg = validate_rag_system()
    if not is_valid:
        return create_error_response(error_msg, 503)
        
    if 'file' not in request.files:
        return create_error_response("No file provided", 400)
        
    file = request.files['file']
    if file.filename == '':
        return create_error_response("No file selected", 400)
        
    if not file or not allowed_file(file.filename):
        return create_error_response("Invalid file format. Only PDF files are allowed.", 400)
        
    try:
        filename = secure_filename(file.filename)
        if not filename:
            return create_error_response("Invalid filename", 400)
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(file_path)
        logger.info(f"Uploaded file saved: {filename}")
        
        # Get file size for logging
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing PDF: {filename} ({file_size} bytes)")
        
        start_time = datetime.now()
        
        # Process PDF
        success = rag_system.update_knowledge_base_from_pdf(file_path)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up uploaded file: {filename}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup file {filename}: {cleanup_error}")
        
        if success:
            # Get updated stats
            stats = rag_system.get_kb_stats()
            
            logger.info(f"Successfully processed PDF: {filename} in {processing_time:.2f}s")
            return create_success_response({
                "filename": filename,
                "file_size_bytes": file_size,
                "processing_time_seconds": round(processing_time, 3),
                "updated_stats": stats
            }, f"Knowledge base updated from PDF: {filename}")
        else:
            return create_error_response("PDF content could not be processed or contained no valid text")
            
    except Exception as e:
        # Clean up file in case of error
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
            
        logger.error(f"Error processing PDF upload: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_error_response(f"Failed to process PDF: {str(e)}")

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get search analytics and usage metrics"""
    is_valid, error_msg = validate_rag_system()
    if not is_valid:
        return create_error_response(error_msg, 503)
    
    if not rag_system.db_manager or not rag_system.db_manager.connection:
        return create_error_response("Database not available for analytics", 503)
    
    try:
        # Basic analytics - you can extend this with more database queries
        analytics_data = {
            "analytics_available": True,
            "note": "Basic analytics ready. Extend DatabaseManager for detailed metrics.",
            "suggested_queries": [
                "SELECT COUNT(*) FROM search_logs WHERE DATE(search_timestamp) = CURDATE()",
                "SELECT AVG(response_time_ms) FROM search_logs WHERE search_timestamp > DATE_SUB(NOW(), INTERVAL 1 DAY)",
                "SELECT query, COUNT(*) as frequency FROM search_logs GROUP BY query ORDER BY frequency DESC LIMIT 10"
            ]
        }
        
        return create_success_response(analytics_data)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return create_error_response(f"Analytics retrieval failed: {str(e)}")

@app.route('/reinitialize', methods=['POST'])
def reinitialize_system():
    """Reinitialize RAG system (admin endpoint)"""
    try:
        logger.info("Reinitializing RAG system...")
        
        # Close existing system if available
        global rag_system
        if rag_system:
            try:
                rag_system.__del__()
            except:
                pass
        
        # Reinitialize
        initialize_rag_system()
        
        if rag_system:
            stats = rag_system.get_kb_stats()
            return create_success_response({
                "reinitialized": True,
                "stats": stats
            }, "RAG system reinitialized successfully")
        else:
            return create_error_response("Failed to reinitialize RAG system", 500)
            
    except Exception as e:
        logger.error(f"Error reinitializing system: {str(e)}")
        return create_error_response(f"Reinitialization failed: {str(e)}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return create_error_response("Endpoint not found", 404)

@app.errorhandler(405)
def method_not_allowed(error):
    return create_error_response("Method not allowed", 405)

@app.errorhandler(413)
def too_large(error):
    return create_error_response("File too large. Maximum size is 16MB.", 413)

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return create_error_response("Internal server error", 500)

@app.before_request
def log_request():
    """Log incoming requests"""
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Enhanced RAG API server")
    logger.info(f"Host: {host}, Port: {port}, Debug: {debug}")
    logger.info(f"RAG System Status: {'Ready' if rag_system else 'Not Ready'}")
    
    app.run(debug=debug, host=host, port=port, threaded=True)