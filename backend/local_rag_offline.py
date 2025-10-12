# local_rag_offline_images.py
"""
Local offline RAG engine: IMAGE-ONLY VERSION
 - Ingests: Images (.png/.jpg/.jpeg/...).
 - OCR for images via EasyOCR.
 - Embeddings with sentence-transformers (all-MiniLM-L6-v2).
 - Index: FAISS if available, else sklearn NearestNeighbors.
 - Persistence: metadata.json + embeddings.npy (+ faiss.index when used).
 - Extractive summarization: sentence embeddings ranked by query relevance.
"""

import os, json, hashlib, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import threading
import traceback

# Text extraction libraries (Removed PyPDF2 and docx)
import easyocr
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# --- External Indexing Library Check ---
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Download NLTK data required for sentence tokenization (must run once)
nltk.download('punkt', quiet=True) 


class LocalRAGOffline:
    def __init__(self, index_dir: str = "./rag_offline_index", model_name: str = "all-MiniLM-L6-v2", use_faiss: bool = True, gpu_ocr: bool = False):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        
        # OCR Reader is essential
        self.reader = easyocr.Reader(['en'], gpu=gpu_ocr)  

        self.use_faiss = use_faiss and _HAS_FAISS
        self.meta_path = self.index_dir / "metadata.json"
        self.emb_path = self.index_dir / "embeddings.npy"
        self.faiss_path = self.index_dir / "faiss.index"

        self.metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

        self._faiss_index = None
        self._nn_index = None

        self._load()

    # ---------------- Utilities (Same) ----------------
    def _hash(self, s: str) -> str:
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        if self.embeddings is not None:
            np.save(self.emb_path, self.embeddings)
        if self.use_faiss and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self.faiss_path))

    def _load(self):
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = []

        if self.emb_path.exists():
            try:
                self.embeddings = np.load(self.emb_path)
            except Exception:
                self.embeddings = None

        if self.use_faiss and self.faiss_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(self.faiss_path))
            except Exception:
                self._faiss_index = None

        if self.embeddings is not None and self._faiss_index is None and not self.use_faiss:
            self._nn_index = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute').fit(self.embeddings)

        if self.embeddings is not None and self.use_faiss and self._faiss_index is None:
            self._build_faiss_index()

    # ---------------- Extraction (Image-Only) ----------------
    
    def _read_image_ocr(self, path: str) -> str:
        """Helper to read text from an image path using EasyOCR."""
        try:
            # Use paragraph=True to get text blocks that are easier to chunk
            res = self.reader.readtext(path, detail=0, paragraph=True)
            return "\n".join([r.strip() for r in res if r and r.strip()])
        except Exception:
            return ""

    def _extract_text(self, path: str) -> str:
        """Only checks for image extensions and calls OCR."""
        ext = Path(path).suffix.lower()
        
        # Only support image file types
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return self._read_image_ocr(path)
            
        # If it's not an image, return empty string or handle as error
        return "" 

    # ---------------- Chunking (Same) ----------------
    def _chunk_text(self, text: str, max_chars: int = 800) -> List[str]:
        # chunk by sentences filling up to max_chars
        sents = sent_tokenize(text)
        chunks = []
        cur = ""
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        if not chunks and text.strip():
            return [text.strip()]
        return chunks

    # ---------------- Index building (Same) ----------------
    def _build_faiss_index(self):
        if self.embeddings is None:
            return
        emb = self.embeddings.astype(np.float32)
        index = faiss.IndexFlatIP(self.dim)  
        index.add(emb)
        self._faiss_index = index
        self._nn_index = None

    def _rebuild_index(self):
        if self.embeddings is None:
            return
        if self.use_faiss:
            self._build_faiss_index()
        else:
            self._nn_index = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute').fit(self.embeddings)
            self._faiss_index = None

    # ---------------- Ingestion (Same Logic Flow) ----------------
    def ingest_file(self, path: str, force: bool = False) -> Tuple[bool, str]:
        """
        Ingest a single image file: extract text, chunk, embed chunks, append to index.
        """
        try:
            text = self._extract_text(path)
            if not text or len(text.strip()) < 20:
                return False, f"No/insufficient text in non-document file: {Path(path).name}"
            
            chunks = self._chunk_text(text)
            added = 0
            new_embs = []
            
            for i, c in enumerate(chunks):
                chunk_id = self._hash(path + f"||{i}||{len(c)}")
                if not force and any(m['id'] == chunk_id for m in self.metadata):
                    continue
                    
                meta = {"id": chunk_id, "source": path, "filename": os.path.basename(path), "text": c, "chunk_index": i}
                self.metadata.append(meta)
                
                # --- Core Deep Learning Embedding ---
                emb = self.embedder.encode(c, convert_to_numpy=True, show_progress_bar=False)
                
                # normalize to unit length for cosine via inner product
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                new_embs.append(emb.reshape(1, -1))
                added += 1
                
            if added:
                new_block = np.vstack(new_embs).astype(np.float32)
                if self.embeddings is None:
                    self.embeddings = new_block
                else:
                    self.embeddings = np.vstack([self.embeddings, new_block])
                
                self._rebuild_index()
                self._save()
                return (added > 0), f"Indexed {added} chunks from {Path(path).name}"
            
            return (added > 0), f"Skipped {Path(path).name}. Already indexed."
                
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"Error ingesting {Path(path).name}: {e}\n{tb}"

    def ingest_folder(self, folder: str, exts: Optional[List[str]] = None, recursive: bool = True) -> List[Tuple[str, str]]:
        """
        Walk folder and ingest files (only supports image extensions).
        """
        # Enforce image extensions only
        exts = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
        folder_p = Path(folder)
        results = []
        
        if recursive:
            files = [str(p) for p in folder_p.rglob("*") if p.suffix.lower() in exts]
        else:
            files = [str(p) for p in folder_p.iterdir() if p.suffix.lower() in exts]
            
        for f in tqdm(files, desc=f"Ingesting {folder}"):
            ok, msg = self.ingest_file(f)
            results.append((f, msg))
        return results

    # ---------------- Search (Same) ----------------
    def _search_bruteforce(self, q_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        # embeddings dot query
        sims = (self.embeddings @ q_emb.reshape(-1, 1)).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in idxs]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.embeddings is None or len(self.metadata) == 0:
            return []
            
        q = self.embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        if np.linalg.norm(q) > 0:
            q = q / np.linalg.norm(q)
        else:
            q = q

        # --- Perform Search based on Index Type ---
        if self.use_faiss and self._faiss_index is not None:
            # FAISS Search
            qf = q.astype(np.float32).reshape(1, -1)
            D, I = self._faiss_index.search(qf, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx == -1: continue
                m = self.metadata[int(idx)]
                results.append({"id": m["id"], "source": m["source"], "filename": m["filename"], "text": m["text"], "score": float(score)})
            return results
            
        elif self._nn_index is not None:
            # Scikit-learn Nearest Neighbors Search
            dists, idxs = self._nn_index.kneighbors(q.reshape(1, -1), n_neighbors=min(top_k, len(self.embeddings)))
            results = []
            for dist, idx in zip(dists[0], idxs[0]):
                score = 1.0 - float(dist)  # Convert distance to similarity
                m = self.metadata[int(idx)]
                results.append({"id": m["id"], "source": m["source"], "filename": m["filename"], "text": m["text"], "score": float(score)})
            return results
            
        else:
            # Brute force search (fallback)
            pairs = self._search_bruteforce(q, top_k=top_k)
            res = []
            for idx, score in pairs:
                m = self.metadata[idx]
                res.append({"id": m["id"], "source": m["source"], "filename": m["filename"], "text": m["text"], "score": float(score)})
            return res

    # ---------------- Summarization / QA (Same Logic) ----------------
    def summarize_document(self, path: str, top_n_sentences: int = 3) -> str:
        """Extractive summarization using centrality."""
        chunks = [m for m in self.metadata if m['source'] == path]
        if not chunks: return "No indexed chunks from this document."
        all_text = " ".join([c['text'] for c in chunks])
        sents = sent_tokenize(all_text)
        if not sents: return all_text[:1000] + "..."
        s_embs = self.embedder.encode(sents, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(s_embs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        s_embs = s_embs / norms
        centroid = np.mean(s_embs, axis=0, keepdims=True)
        sims = (s_embs @ centroid.T).squeeze()
        top_idx = np.argsort(-sims)[:top_n_sentences]
        summary = " ".join([sents[i] for i in sorted(top_idx)])
        return summary

    def answer_query(self, query: str, top_k: int = 5, summary_sentences:int = 2) -> Dict:
        """
        Retrieves top_k chunks and builds an extractive answer by ranking sentences
        from top docs by similarity to query.
        """
        hits = self.search(query, top_k=top_k)
        if not hits:
            return {"query": query, "answer": "No relevant documents found.", "matches": []}
        
        # 1. Collect candidate sentences
        candidate_sents = []
        for h in hits:
            candidate_sents.extend(sent_tokenize(h['text'])) 

        s_texts = [s for s in candidate_sents if s.strip()]
        if not s_texts:
            return {"query": query, "answer": hits[0]['text'][:1000] + "...", "matches": hits}
        
        # 2. Rank sentences by similarity to query embedding
        s_embs = self.embedder.encode(s_texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(s_embs, axis=1, keepdims=True); norms[norms==0] = 1.0
        s_embs = s_embs / norms
        
        q_emb = self.embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        if np.linalg.norm(q_emb) > 0: q_emb = q_emb / np.linalg.norm(q_emb)
        
        sims = (s_embs @ q_emb.reshape(-1,1)).squeeze()
        top_ids = np.argsort(-sims)[:summary_sentences]
        answer = " ".join([s_texts[i] for i in sorted(top_ids)])

        return {"query": query, "answer": answer, "matches": hits}

    # ---------------- Helpers for UI (Same) ----------------
    def list_indexed_files(self) -> List[str]:
        files = sorted(list({m['source'] for m in self.metadata}))
        return files

    def get_document_text(self, path: str) -> str:
        parts = [m['text'] for m in self.metadata if m['source'] == path]
        return "\n\n".join(parts) if parts else ""