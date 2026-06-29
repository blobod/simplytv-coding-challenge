"""FastAPI search backend.

Loads the precomputed embeddings + filenames once at startup and answers text
queries with exact cosine similarity (a dot product, since everything is
L2-normalized). Thumbnails are served straight from disk as static files.

Run: uvicorn app:app --reload
"""

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = Path("embeddings.npy")
INDEX_PATH = Path("index.json")
THUMBS_DIR = Path("thumbnails")
MODEL_NAME = "clip-ViT-B-32"
DEFAULT_K = 12

app = FastAPI(title="Semantic Image Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev: any frontend origin may call us
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at import time and kept in memory for the process lifetime.
model = SentenceTransformer(MODEL_NAME)
embeddings = np.load(EMBEDDINGS_PATH)
filenames = json.loads(INDEX_PATH.read_text())["filenames"]

app.mount("/thumbnails", StaticFiles(directory=THUMBS_DIR), name="thumbnails")


@app.get("/")
def home():
    return FileResponse("index.html")


class SearchRequest(BaseModel):
    query: str
    k: int = DEFAULT_K


@app.post("/search")
def search(req: SearchRequest):
    query_vec = model.encode(
        req.query, convert_to_numpy=True, normalize_embeddings=True
    )

    # Cosine similarity == dot product because both sides are L2-normalized.
    scores = embeddings @ query_vec

    k = min(req.k, len(filenames))
    top = np.argsort(-scores)[:k]

    results = [
        {
            "filename": filenames[i],
            # thumbnails are saved as "<stem>.jpg" by the indexer
            "thumbnail": f"{Path(filenames[i]).stem}.jpg",
            "score": float(scores[i]),
        }
        for i in top
    ]
    return {"query": req.query, "results": results}
