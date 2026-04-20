import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load cleaned books ──────────────────────────────────────────
print("Loading books...")
books = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))
books["content"] = books["content"].fillna("")
print(f"  Books shape: {books.shape}")

# ── Build TF-IDF matrix ─────────────────────────────────────────
print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)   # unigrams + bigrams
)
tfidf_matrix = tfidf.fit_transform(books["content"])
print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

# ── Save vectorizer and matrix ──────────────────────────────────
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "wb") as f:
    pickle.dump(tfidf_matrix, f)

# Save index mapping: isbn -> row position
isbn_to_idx = pd.Series(books.index, index=books["isbn"]).to_dict()
title_to_idx = pd.Series(books.index, index=books["title"].str.lower()).to_dict()

with open(os.path.join(MODEL_DIR, "isbn_to_idx.pkl"), "wb") as f:
    pickle.dump(isbn_to_idx, f)

with open(os.path.join(MODEL_DIR, "title_to_idx.pkl"), "wb") as f:
    pickle.dump(title_to_idx, f)

print("Saved TF-IDF model files to models/")
print("Content-based model done!")


# ── Helper: get similar books by title ─────────────────────────
def get_content_recommendations(book_title, books, tfidf_matrix, title_to_idx, n=10):
    """Return top-N books most similar to book_title using cosine similarity."""
    title_lower = book_title.lower()

    # Find closest matching title
    match = None
    for t in title_to_idx:
        if title_lower in t or t in title_lower:
            match = t
            break

    if match is None:
        print(f"  Book '{book_title}' not found in dataset.")
        return pd.DataFrame()

    idx = title_to_idx[match]

    # Compute cosine similarity between this book and all others
    query_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Sort by similarity, exclude itself
    sim_indices = np.argsort(sim_scores)[::-1]
    sim_indices = [i for i in sim_indices if i != idx][:n]

    result = books.iloc[sim_indices][["isbn", "title", "author"]].copy()
    result["similarity"] = sim_scores[sim_indices]
    return result.reset_index(drop=True)


if __name__ == "__main__":
    sample_title = books["title"].iloc[0]
    print(f"\nBooks similar to: '{sample_title}'")
    recs = get_content_recommendations(
        sample_title, books, tfidf_matrix, title_to_idx, n=5
    )
    if not recs.empty:
        print(recs[["title", "author", "similarity"]].to_string(index=False))
