# Semantic Image Search

Type a query in plain English and get back the images from a folder, ranked by how well they match.

## How it works

The app is built on CLIP (`clip-ViT-B-32`, via `sentence-transformers`), a model that embeds both images and text into one shared vector space. Because pictures and words live in the same space, an image of a dog on a beach and the phrase *"a dog on the beach"* end up close together.

That lets the work split cleanly into two phases. Offline, every image in the folder gets embedded once and the vectors are saved to disk. Then at search time, only the short text query gets embedded and compared against those precomputed image vectors.

Ranking is by cosine similarity between the query vector and each image vector. The `score` you get back is a similarity value, not a probability. It tells you how aligned two vectors are, and it only really means something relative to the other scores in the same set of results.

## Flow

```
INDEX (offline, run once per image set)
  images/ ──▶ CLIP image encoder ──▶ embeddings.npy + index.json (filenames)
        └───▶ Pillow ──▶ thumbnails/ (small JPEGs)

SEARCH (per query)
  text query ──▶ CLIP text encoder ──▶ query vector
                                          │
        embeddings.npy (held in memory) ──┴─▶ cosine similarity ──▶ top-k ──▶ frontend grid
```

## Design decisions

**Exact NumPy cosine similarity, no vector database.** A folder holds a few thousand vectors at most. At that size an exact dot product over the in memory matrix runs in milliseconds, and the whole search is a single `embeddings @ query_vec`. An approximate nearest neighbour index or a dedicated vector store would add moving parts and operational cost, and it would actually hurt recall (ANN is approximate) for no real speed gain here. At millions of images the tradeoff flips and I'd reach for ANN, a proper vector store, and a thumbnail CDN. Being clear about where that line sits is the whole point.

**Embeddings are precomputed offline, not per query.** Embedding an image is the slow part, so it happens once at index time. After that, search never runs the model on images at all. It just embeds the short query string and does a matrix multiply. The only time you need to reindex is when the set of images changes.

**Low latency and cheap serving.** The embedding matrix is loaded into memory once at startup and reused for every request. Images go out as small static JPEG thumbnails built at index time, so nothing gets processed per request. Each query only embeds a short string and multiplies it against a matrix that's already sitting in RAM.

**About the scores.** CLIP cosine values are low in absolute terms, often well under 0.4. That's just the shape of the embedding space, not a sign of a weak match. What counts is the ranking, the relative order of the scores rather than how big they are.

## Running it

The repo comes with a set of sample images already sitting in `images/`, so there's nothing to add by hand. From the project root, just install, index, and serve:

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. build the index (embeddings + thumbnails) from images/
python index.py

# 3. start the server
uvicorn app:app --reload

# 4. open the page
#    http://localhost:8000
```

`index.py` reads everything in `images/` and writes out `embeddings.npy`, `index.json`, and the `thumbnails/` folder. `app.py` loads those at startup and serves both the API (`POST /search`) and the frontend (`index.html`) on the same port. If you ever want to refresh the sample set, `python fetch_samples.py` repopulates `images/`.

## Using your own images

Want to search your own photos instead? Drop them straight into the `images/` folder (JPG, PNG, WebP, BMP, and GIF all work). You can clear out the samples first or just add alongside them.

Then reindex so the new images get embedded and thumbnailed:

```bash
python index.py
```

That rebuilds `embeddings.npy`, `index.json`, and `thumbnails/` from whatever is currently in `images/`. Restart the server (or rely on `--reload`) and your own images are searchable. Anything you add to `images/` only shows up after you reindex, since search reads from the saved vectors, never from the folder directly.

## Limitations

CLIP is a general purpose model, and it has some known blind spots:

- **Text inside images** (signs, documents, logos). It reads scenes, not words.
- **Exact counts.** "Three cats" and "five cats" look about the same to it.
- **Fine grained attributes.** Precise colors, small details, and subtle distinctions are unreliable.
- **Concept coverage.** It only knows what was in its training data, so novel or niche subjects may not match well.

## Core questions

**Why CLIP?** It embeds images and text into one shared vector space, so a text query can be compared directly against images, which is exactly what natural language image search needs. It also runs locally and free, with no API keys and no per query cost.

**Why no vector database?** A folder is only a few thousand vectors, and an exact NumPy dot product over them runs in milliseconds. A vector store or ANN index would add complexity and cost recall for no benefit at this scale. I'd only reach for one at millions of images.

**How does it stay low latency and cheap?** Image embeddings are computed once offline and kept in memory, so each query just embeds a short string and does one matrix multiply. Images are served as small static thumbnails built at index time, with no per request processing.

**What does the similarity score mean?** It's the cosine similarity between the query and image vectors, a measure of alignment rather than a probability. The values run low by the nature of CLIP's space, so what matters is the ranking, not the magnitude.

**What are the limitations?** CLIP is weak on text inside images, exact counts, and fine grained attributes, and it only knows concepts that were in its training data.

## Notes

`tf-keras` is pinned for `transformers` and Keras 3 compatibility.
