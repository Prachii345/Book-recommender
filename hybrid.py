import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = "data"
MODEL_DIR = "models"

# ── Load all models and data ────────────────────────────────────
print("Loading models and data...")

df       = pd.read_csv(os.path.join(DATA_DIR, "ratings_clean.csv"))
books_df = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))
books_df["content"] = books_df["content"].fillna("")

with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
    svd_data = pickle.load(f)

with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb") as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "title_to_idx.pkl"), "rb") as f:
    title_to_idx = pickle.load(f)

with open(os.path.join(MODEL_DIR, "book_graph.pkl"), "rb") as f:
    G = pickle.load(f)

print("All models loaded!")


# ── Content-based helper ────────────────────────────────────────
def get_content_scores(book_title, books_df, tfidf_matrix, title_to_idx, n=20):
    title_lower = book_title.lower()
    match = None
    for t in title_to_idx:
        if title_lower in t or t in title_lower:
            match = t
            break
    if match is None:
        return {}
    idx = title_to_idx[match]
    query_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idxs   = np.argsort(sim_scores)[::-1][1:n+1]
    return {books_df.iloc[i]["isbn"]: float(sim_scores[i]) for i in top_idxs}


# ── SVD helper ──────────────────────────────────────────────────
def get_svd_scores(user_id, df, svd_data, n=20):
    user_enc     = svd_data["user_enc"]
    book_enc     = svd_data["book_enc"]
    book_dec     = svd_data["book_dec"]
    user_factors = svd_data["user_factors"]
    item_factors = svd_data["item_factors"]

    if user_id not in user_enc:
        return {}

    u_idx  = user_enc[user_id]
    u_vec  = user_factors[u_idx]
    scores = item_factors.T.dot(u_vec)

    rated_isbns = set(df[df["user_id"] == user_id]["isbn"].tolist())
    rated_idxs  = {book_enc[i] for i in rated_isbns if i in book_enc}

    ranked   = np.argsort(scores)[::-1]
    top_idxs = [i for i in ranked if i not in rated_idxs][:n]

    top_scores = np.array([scores[i] for i in top_idxs])
    if top_scores.max() != top_scores.min():
        top_scores = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())

    return {book_dec[i]: float(top_scores[j]) for j, i in enumerate(top_idxs)}


# ── A* graph helper ─────────────────────────────────────────────
def get_graph_scores(start_isbn, G, n=20):
    if start_isbn not in G:
        return {}
    visited  = set()
    result   = {}
    frontier = list(G.neighbors(start_isbn))
    depth    = 1.0

    while frontier and len(result) < n:
        next_frontier = []
        for node in frontier:
            if node not in visited and node != start_isbn:
                visited.add(node)
                weights = [G[node][nb]["weight"] for nb in G.neighbors(node) if nb in visited or nb == start_isbn]
                score   = (np.mean(weights) if weights else 0.1) / depth
                result[node] = float(score)
                next_frontier.extend([nb for nb in G.neighbors(node) if nb not in visited])
        frontier = next_frontier
        depth   += 1.0

    if result:
        vals = np.array(list(result.values()))
        if vals.max() != vals.min():
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        result = dict(zip(result.keys(), vals.tolist()))
    return result


# ── Hybrid recommender ──────────────────────────────────────────
def hybrid_recommend(user_id, liked_book_title, df, books_df,
                     svd_data, tfidf_matrix, title_to_idx, G,
                     alpha=0.4, beta=0.4, gamma=0.2, n=10):
    """
    Combine SVD + TF-IDF + A* with weighted scores.
    alpha = SVD weight (collaborative)
    beta  = TF-IDF weight (content-based)
    gamma = A* graph weight (diversity)
    """
    # Get liked book ISBN for graph search
    title_lower = liked_book_title.lower()
    liked_isbn  = None
    for t, idx in title_to_idx.items():
        if title_lower in t or t in title_lower:
            liked_isbn = books_df.iloc[idx]["isbn"]
            break

    # Get scores from each model
    svd_scores     = get_svd_scores(user_id, df, svd_data, n=30)
    content_scores = get_content_scores(liked_book_title, books_df, tfidf_matrix, title_to_idx, n=30)
    graph_scores   = get_graph_scores(liked_isbn, G, n=30) if liked_isbn else {}

    # Combine all candidate ISBNs
    all_isbns = set(svd_scores) | set(content_scores) | set(graph_scores)

    # Compute weighted hybrid score
    hybrid_scores = {}
    for isbn in all_isbns:
        score = (alpha * svd_scores.get(isbn, 0) +
                 beta  * content_scores.get(isbn, 0) +
                 gamma * graph_scores.get(isbn, 0))
        hybrid_scores[isbn] = score

    # Sort and get top-N
    top_isbns = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:n]

    result = books_df[books_df["isbn"].isin(top_isbns)][["isbn","title","author"]].copy()
    result["hybrid_score"] = result["isbn"].map(hybrid_scores)
    result["svd_score"]    = result["isbn"].map(svd_scores)
    result["content_score"]= result["isbn"].map(content_scores)
    result["graph_score"]  = result["isbn"].map(graph_scores)
    result = result.fillna(0).sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    return result


# ── Evaluation metrics ──────────────────────────────────────────
def evaluate_recommendations(df, svd_data, books_df, n_users=50, k=10):
    """Compute Precision@K and Recall@K on a sample of users."""
    print("\nEvaluating recommendation quality...")

    user_enc = svd_data["user_enc"]
    users    = [u for u in df["user_id"].unique()[:n_users] if u in user_enc]

    precisions, recalls = [], []

    for user_id in users:
        # Ground truth: books rated >= 7 (liked)
        user_ratings = df[df["user_id"] == user_id]
        liked = set(user_ratings[user_ratings["rating"] >= 7]["isbn"].tolist())
        if len(liked) == 0:
            continue

        # Get SVD recommendations
        scores  = get_svd_scores(user_id, df, svd_data, n=k)
        rec_set = set(scores.keys())

        hits      = len(rec_set & liked)
        precision = hits / k
        recall    = hits / len(liked) if liked else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall    = np.mean(recalls) if recalls else 0

    print(f"  Precision@{k} : {avg_precision:.4f}")
    print(f"  Recall@{k}    : {avg_recall:.4f}")
    return avg_precision, avg_recall


# ── Main demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Evaluation
    evaluate_recommendations(df, svd_data, books_df, n_users=100, k=10)

    # Demo hybrid recommendation
    sample_user  = df["user_id"].iloc[0]
    sample_book  = "The Da Vinci Code"

    recs = hybrid_recommend(
        user_id         = sample_user,
        liked_book_title= sample_book,
        df              = df,
        books_df        = books_df,
        svd_data        = svd_data,
        tfidf_matrix    = tfidf_matrix,
        title_to_idx    = title_to_idx,
        G               = G,
        alpha=0.4, beta=0.4, gamma=0.2,
        n=10
    )

    print(f"\nHybrid recommendations for user {sample_user}")
    print(f"Based on liking: '{sample_book}'")
    print(recs[["title","author","hybrid_score"]].to_string(index=False))
    print("\nHybrid fusion done!")
