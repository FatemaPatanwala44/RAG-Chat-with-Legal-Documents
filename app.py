import os
import shutil
import pickle
import uvicorn
import nltk
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
import faiss
from rank_bm25 import BM25Okapi

# ==================== GOOGLE EMBEDDINGS CLASS ====================
# Add this import at the top of your file
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

class SimpleEmbeddings:
    def __init__(self, dim=384):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words='english')
        self.is_fitted = False
        self.all_texts = []
        print("✅ Using Simple Embeddings (no API required)")
    
    def embed_query(self, text):
        """Embed a single query text"""
        if not self.is_fitted or not self.all_texts:
            # If not fitted, return random but deterministic embedding
            return self._deterministic_embedding(text)
        
        # Transform the query using fitted vectorizer
        vec = self.vectorizer.transform([text]).toarray()[0]
        return vec.tolist()
    
    def embed_documents(self, texts):
        """Embed multiple documents"""
        self.all_texts = texts
        
        # Fit the vectorizer on all texts
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
        # Transform all texts
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()
    
    def _deterministic_embedding(self, text):
        """Create a deterministic embedding based on text hash (fallback)"""
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        return np.random.randn(self.dim).tolist()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== CONFIG ====================
UPLOAD_DIR = "docs"
INDEX_DIR = "legal_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Legal keywords for classification
LEGAL_KEYWORDS = [
    "act", "section", "article", "clause", "regulation", "statute",
    "court", "judgment", "plaintiff", "defendant", "legal", "law",
    "contract", "agreement", "breach", "damages", "indemnity",
    "witness", "whereas", "now therefore", "party", "liability",
    "shall", "hereby", "hereinafter", "aforesaid", "therein"
]

# ==================== INITIALIZE MODELS ====================
print("Loading embedding model...")
# USING GOOGLE EMBEDDINGS INSTEAD OF HUGGINGFACE
# Use SimpleEmbeddings (no API calls, works offline)
embedding_model = SimpleEmbeddings(dim=384)

# OR use even simpler WordCountEmbeddings
# embedding_model = WordCountEmbeddings()

print("Loading LLM...")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Global variables for search
chunks = []
chunk_metadata = []
embeddings = None
bm25 = None
faiss_index = None
current_file = None

# ==================== HELPER FUNCTIONS ====================
def is_legal_document(text, filename=""):
    """Classify if document is legal-related (with lower threshold)"""
    text_lower = text.lower()
    
    # Count keyword matches
    matches = 0
    matched_keywords = []
    for kw in LEGAL_KEYWORDS:
        if kw in text_lower:
            matches += 1
            matched_keywords.append(kw)
    
    # REDUCED CRITERIA: Lower the expectation
    # Before: matches / 15 * 100
    # After: matches / 5 * 100 (only need 5 keywords for 100%)
    confidence = min(100, (matches / 5) * 100)  # CHANGED FROM 15 TO 5
    
    # Check filename for legal hints (bonus)
    filename_lower = filename.lower()
    legal_file_terms = ["legal", "contract", "agreement", "case","resume","cv","certifications", 
                        "syllabus", "curriculum", "course", "university", "college", "academic", 
                        "b.tech", "btech", "engineering", "scheme", "regulation"]  # ADDED MORE TERMS
    if any(term in filename_lower for term in legal_file_terms):
        confidence += 15  # INCREASED BONUS FROM 10 TO 15
    
    # LOWERED THRESHOLD: From 30 to 15
    is_legal = confidence > 15  # CHANGED FROM 30 TO 15
    
    return {
        "is_legal": is_legal,
        "confidence": round(min(100, confidence), 2),
        "keywords_found": matched_keywords[:10],
        "total_matches": matches
    }

def chunk_document(text, metadata):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = splitter.split_text(text)
    
    chunks_data = []
    for i, chunk_text in enumerate(texts):
        # Skip empty chunks
        if not chunk_text.strip():
            continue
            
        chunks_data.append({
            "text": chunk_text,
            "metadata": {
                **metadata,
                "chunk_id": i,
                "page": i // 2 + 1  # Approximate page number
            }
        })
    return chunks_data

def build_index(chunks_data):
    """Create FAISS and BM25 indexes"""
    global chunks, chunk_metadata, embeddings, bm25, faiss_index
    
    if not chunks_data:
        return False
    
    chunks = [c["text"] for c in chunks_data]
    chunk_metadata = [c["metadata"] for c in chunks_data]
    
    print(f"Building index for {len(chunks)} chunks...")
    
    # Create dense embeddings using Google
    print("Generating embeddings with Google...")
    embeddings_list = embedding_model.embed_documents(chunks)
    embeddings = np.array(embeddings_list)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype('float32'))
    
    # Build BM25 index
    tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Save to disk
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({
            "chunks": chunks, 
            "metadata": chunk_metadata, 
            "embeddings": embeddings
        }, f)
    
    return True

def load_index():
    """Load existing index from disk"""
    global chunks, chunk_metadata, embeddings, bm25, faiss_index
    
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                chunks = data["chunks"]
                chunk_metadata = data["metadata"]
                embeddings = data["embeddings"]
            
            # Rebuild BM25 index
            tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk in chunks]
            bm25 = BM25Okapi(tokenized_chunks)
            
            return True
        except:
            return False
    return False

def hybrid_search(query, k=5):
    """Combine dense and sparse search"""
    if faiss_index is None or bm25 is None or not chunks:
        return []
    
    # Dense search using Google
    query_emb = embedding_model.embed_query(query)
    query_emb = np.array([query_emb])
    distances, indices = faiss_index.search(query_emb.astype('float32'), k*2)
    dense_indices = indices[0]
    
    # Sparse search (BM25)
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top sparse indices
    sparse_indices = np.argsort(bm25_scores)[::-1][:k*2]
    
    # Combine scores
    combined_scores = {}
    
    # Dense: weight 0.7
    for rank, idx in enumerate(dense_indices):
        if idx < len(chunks):
            combined_scores[idx] = combined_scores.get(idx, 0) + (1.0 / (rank + 1)) * 0.7
    
    # Sparse: weight 0.3
    for rank, idx in enumerate(sparse_indices):
        if idx < len(chunks):
            combined_scores[idx] = combined_scores.get(idx, 0) + (bm25_scores[idx] / max(bm25_scores)) * 0.3
    
    # Sort by combined score
    top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
    
    results = []
    for idx in top_indices:
        if idx < len(chunks):
            results.append((chunks[idx], chunk_metadata[idx]))
    
    return results

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": len(chunks) > 0}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_file
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    # Check file size (50MB limit)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")
    
    # Save temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load PDF
        print(f"Loading PDF: {file.filename}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        if not pages:
            raise HTTPException(400, "PDF has no readable text")
        
        full_text = "\n".join([p.page_content for p in pages])
        
        # Check if document has content
        if len(full_text.strip()) < 100:
            raise HTTPException(400, "PDF has insufficient text content")
        
        # Legal classification
        print("Classifying document...")
        classification = is_legal_document(full_text, file.filename)
        
        if not classification["is_legal"]:
            os.remove(file_path)
            raise HTTPException(400, f"Not a legal document. Confidence: {classification['confidence']}%. Found {classification['total_matches']} legal keywords.")
        
        # Create metadata
        metadata = {
            "source": file.filename,
            "total_pages": len(pages)
        }
        
        # Chunk document
        print("Chunking document...")
        chunks_data = chunk_document(full_text, metadata)
        
        # Build index
        print("Building search index...")
        success = build_index(chunks_data)
        
        if not success:
            raise HTTPException(500, "Failed to build index")
        
        # Cleanup temp file
        os.remove(file_path)
        current_file = file.filename
        
        return {
            "message": f"✅ {file.filename} uploaded and indexed successfully",
            "chunks": len(chunks_data),
            "pages": len(pages),
            "classification": classification
        }
        
    except Exception as e:
        # Cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/chat")
async def chat(query: str = Form(...)):
    if not chunks:
        # Try to load existing index
        if not load_index():
            raise HTTPException(400, "No document uploaded yet. Please upload a legal PDF first.")
    
    # Validate query
    if len(query.strip()) < 3:
        raise HTTPException(400, "Query too short")
    
    # Hybrid search
    results = hybrid_search(query, k=5)
    
    if not results:
        return {
            "answer": "I couldn't find any relevant information in the document to answer your question.",
            "sources": []
        }
    
    # Build context
    context_parts = []
    sources = []
    
    for chunk_text, meta in results:
        page = meta.get('page', '?')
        source = meta.get('source', 'Unknown')
        context_parts.append(f"[From page {page} of {source}]:\n{chunk_text}")
        if page not in sources and page != '?':
            sources.append(page)
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a legal assistant. Answer the question based STRICTLY on the following context from legal documents. 
If the answer cannot be found in the context, say "I cannot find this information in the provided document."
Always cite the page number when referencing specific information.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with page citations):"""
    
    try:
        # Generate answer
        response = client.models.generate_content(
            model="gemini-1.5-flash-8b",
            contents=prompt
        )
        answer = response.text
        
        # Sort sources
        sources.sort()
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return {
            "answer": "I encountered an error generating the response. Please try again.",
            "sources": sources
        }

if __name__ == "__main__":
    # Try to load existing index on startup
    if load_index():
        print(f"✅ Loaded existing index with {len(chunks)} chunks")
    else:
        print("📄 No existing index found. Ready for new uploads.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)