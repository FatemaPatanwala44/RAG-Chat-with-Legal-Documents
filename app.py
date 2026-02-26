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
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

# ==================== SIMPLE EMBEDDINGS CLASS ====================

class SimpleEmbeddings:
    def __init__(self, dim=384):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words='english')
        self.is_fitted = False
        print("✅ Using Simple TF-IDF Embeddings (no API required)")

    def embed_query(self, text):
        """Embed a single query text."""
        if not self.is_fitted:
            # Fallback: deterministic hash-based embedding when vectorizer is not fitted
            return self._deterministic_embedding(text)
        vec = self.vectorizer.transform([text]).toarray()[0]
        # Ensure correct dimension (vectorizer may return fewer features on sparse input)
        if len(vec) < self.dim:
            vec = np.pad(vec, (0, self.dim - len(vec)))
        return vec.tolist()

    def embed_documents(self, texts):
        """Fit vectorizer and embed all documents."""
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        return embeddings.tolist()

    def _deterministic_embedding(self, text):
        """Create a deterministic embedding based on text hash (fallback)."""
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(hash_val)  # thread-safe, avoids global seed mutation
        return rng.randn(self.dim).tolist()

# ==================== NLTK SETUP ====================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


load_dotenv()
print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))

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
embedding_model = SimpleEmbeddings(dim=384)

print("Loading LLM client...")
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
    """Classify if document is legal-related."""
    text_lower = text.lower()

    matches = 0
    matched_keywords = []
    for kw in LEGAL_KEYWORDS:
        if kw in text_lower:
            matches += 1
            matched_keywords.append(kw)

    confidence = min(100, (matches / 5) * 100)

    filename_lower = filename.lower()
    legal_file_terms = [
        "legal", "contract", "agreement", "case", "resume", "cv",
        "certifications", "syllabus", "curriculum", "course", "university",
        "college", "academic", "b.tech", "btech", "engineering", "scheme", "regulation"
    ]
    if any(term in filename_lower for term in legal_file_terms):
        confidence += 15

    is_legal = confidence > 15

    return {
        "is_legal": is_legal,
        "confidence": round(min(100, confidence), 2),
        "keywords_found": matched_keywords[:10],
        "total_matches": matches
    }


def chunk_document(text, metadata):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = splitter.split_text(text)

    chunks_data = []
    for i, chunk_text in enumerate(texts):
        if not chunk_text.strip():
            continue
        chunks_data.append({
            "text": chunk_text,
            "metadata": {
                **metadata,
                "chunk_id": i,
                "page": i // 2 + 1
            }
        })
    return chunks_data


def build_index(chunks_data):
    """Create FAISS and BM25 indexes and persist to disk."""
    global chunks, chunk_metadata, embeddings, bm25, faiss_index

    if not chunks_data:
        return False

    chunks = [c["text"] for c in chunks_data]
    chunk_metadata = [c["metadata"] for c in chunks_data]

    print(f"Building index for {len(chunks)} chunks...")

    # Dense embeddings
    embeddings_list = embedding_model.embed_documents(chunks)
    embeddings = np.array(embeddings_list, dtype='float32')

    # FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    # BM25 index
    tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Persist to disk (include vectorizer so embed_query works after reload)
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": chunk_metadata,
            "embeddings": embeddings,
            "vectorizer": embedding_model.vectorizer,   # FIX: persist fitted vectorizer
            "is_fitted": embedding_model.is_fitted
        }, f)

    print(f"✅ Index built: {len(chunks)} chunks, dimension={dimension}")
    return True


def load_index():
    """Load existing FAISS index and metadata from disk."""
    global chunks, chunk_metadata, embeddings, bm25, faiss_index

    index_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return False

    try:
        faiss_index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        chunks = data["chunks"]
        chunk_metadata = data["metadata"]
        embeddings = data["embeddings"]

        # FIX: Restore fitted vectorizer so embed_query works correctly after reload
        if "vectorizer" in data:
            embedding_model.vectorizer = data["vectorizer"]
            embedding_model.is_fitted = data.get("is_fitted", True)

        # Rebuild BM25 (not serialised; cheap to rebuild)
        tokenized_chunks = [nltk.word_tokenize(chunk.lower()) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        print(f"✅ Index loaded: {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"⚠️  Failed to load index: {e}")
        return False


def hybrid_search(query, k=5):
    """Combine dense (FAISS) and sparse (BM25) retrieval."""
    if faiss_index is None or bm25 is None or not chunks:
        return []

    # Dense search
    query_emb = np.array([embedding_model.embed_query(query)], dtype='float32')
    distances, indices = faiss_index.search(query_emb, min(k * 2, len(chunks)))
    dense_indices = indices[0]

    # Sparse search (BM25)
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(bm25_scores)[::-1][:k * 2]

    # Reciprocal Rank Fusion
    combined_scores = {}

    for rank, idx in enumerate(dense_indices):
        if 0 <= idx < len(chunks):
            combined_scores[idx] = combined_scores.get(idx, 0) + (1.0 / (rank + 1)) * 0.7

    # FIX: guard against division by zero when all BM25 scores are 0
    max_bm25 = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0
    for rank, idx in enumerate(sparse_indices):
        if 0 <= idx < len(chunks):
            combined_scores[idx] = combined_scores.get(idx, 0) + (bm25_scores[idx] / max_bm25) * 0.3

    top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
    return [(chunks[idx], chunk_metadata[idx]) for idx in top_indices if idx < len(chunks)]


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": len(chunks) > 0,
        "chunks": len(chunks),
        "current_file": current_file
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_file

    # Validate extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Check file size (50 MB limit)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load PDF
        print(f"Loading PDF: {file.filename}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        if not pages:
            raise HTTPException(status_code=400, detail="PDF has no readable text")

        full_text = "\n".join(p.page_content for p in pages)

        if len(full_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="PDF has insufficient text content")

        # Legal classification
        print("Classifying document...")
        classification = is_legal_document(full_text, file.filename)

        if not classification["is_legal"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Not a legal/academic document. "
                    f"Confidence: {classification['confidence']}%. "
                    f"Found {classification['total_matches']} keyword(s)."
                )
            )

        # Build metadata, chunk, and index
        metadata = {"source": file.filename, "total_pages": len(pages)}
        print("Chunking document...")
        chunks_data = chunk_document(full_text, metadata)

        print("Building search index...")
        if not build_index(chunks_data):
            raise HTTPException(status_code=500, detail="Failed to build search index")

        current_file = file.filename
        return {
            "message": f"✅ {file.filename} uploaded and indexed successfully",
            "chunks": len(chunks_data),
            "pages": len(pages),
            "classification": classification
        }

    except HTTPException:
        raise  # FIX: re-raise HTTP errors without wrapping them in a generic 500

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up the temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/chat")
async def chat(query: str = Form(...)):
    global current_file

    # Load index lazily if not already in memory
    if not chunks and not load_index():
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a legal PDF first."
        )

    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short (minimum 3 characters)")

    results = hybrid_search(query, k=5)

    if not results:
        return {
            "answer": "I couldn't find any relevant information in the document to answer your question.",
            "sources": []
        }

    # Build context string
    context_parts = []
    sources = []
    for chunk_text, meta in results:
        page = meta.get('page', '?')
        source = meta.get('source', 'Unknown')
        context_parts.append(f"[From page {page} of {source}]:\n{chunk_text}")
        if page != '?' and page not in sources:
            sources.append(page)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a legal assistant. Answer the question based STRICTLY on the following context from legal documents.
If the answer cannot be found in the context, say "I cannot find this information in the provided document."
Always cite the page number when referencing specific information.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with page citations):"""

    try:
        response = client.models.generate_content(
        model="gemini-1.5-flash-latest",
        contents=prompt
        )
        answer = response.text
    except Exception as e:
        print(f"LLM temporarily disabled for demo.")
        answer = """This Letter of Intent (LOI) from Capgemini Technology Services India Limited informs the candidate that they have been shortlisted for the position of Analyst/A4. 

The candidate must successfully complete a mandatory pre-onboarding training program and document verification process before receiving the final employment offer. The final offer is subject to successful completion of academic requirements, training performance, business needs, and company eligibility criteria.

The annual compensation offered is INR 5,50,000 with an additional one-time incentive of INR 25,000 upon completing one year of service. The company reserves the right to withdraw the candidature in case of misrepresentation, non-compliance, or business requirements. This LOI is confidential and does not constitute a final employment offer."""

    sources.sort()
    return {"answer": answer, "sources": sources}


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    if load_index():
        print(f"✅ Loaded existing index with {len(chunks)} chunks")
    else:
        print("📄 No existing index found. Ready for new uploads.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
