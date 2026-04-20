import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = "data"
MODEL_DIR = "models"

st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

# ── Load all models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    df       = pd.read_csv(os.path.join(DATA_DIR, "ratings_clean.csv"))
    books_df = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))
    books_df["content"] = books_df["content"].fillna("")

    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
        svd_data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "title_to_idx.pkl"), "rb") as f:
        title_to_idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "book_graph.pkl"), "rb") as f:
        G = pickle.load(f)

    return df, books_df, svd_data, tfidf_matrix, title_to_idx, G


# ── Model helpers ───────────────────────────────────────────────
def get_content_scores(book_title, books_df, tfidf_matrix, title_to_idx, n=20):
    title_lower = book_title.lower()
    match = None
    for t in title_to_idx:
        if title_lower in t or t in title_lower:
            match = t
            break
    if match is None:
        return {}
    idx        = title_to_idx[match]
    query_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idxs   = np.argsort(sim_scores)[::-1][1:n+1]
    return {books_df.iloc[i]["isbn"]: float(sim_scores[i]) for i in top_idxs}


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
    ranked      = np.argsort(scores)[::-1]
    top_idxs    = [i for i in ranked if i not in rated_idxs][:n]

    top_scores = np.array([scores[i] for i in top_idxs])
    if top_scores.max() != top_scores.min():
        top_scores = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())

    return {book_dec[i]: float(top_scores[j]) for j, i in enumerate(top_idxs)}


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


def hybrid_recommend(user_id, liked_book_title, df, books_df,
                     svd_data, tfidf_matrix, title_to_idx, G,
                     alpha=0.4, beta=0.4, gamma=0.2, n=10):
    title_lower = liked_book_title.lower()
    liked_isbn  = None
    for t, idx in title_to_idx.items():
        if title_lower in t or t in title_lower:
            liked_isbn = books_df.iloc[idx]["isbn"]
            break

    svd_scores     = get_svd_scores(user_id, df, svd_data, n=30)
    content_scores = get_content_scores(liked_book_title, books_df, tfidf_matrix, title_to_idx, n=30)
    graph_scores   = get_graph_scores(liked_isbn, G, n=30) if liked_isbn else {}

    all_isbns = set(svd_scores) | set(content_scores) | set(graph_scores)

    hybrid_scores = {}
    for isbn in all_isbns:
        score = (alpha * svd_scores.get(isbn, 0) +
                 beta  * content_scores.get(isbn, 0) +
                 gamma * graph_scores.get(isbn, 0))
        hybrid_scores[isbn] = score

    top_isbns = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:n]
    result = books_df[books_df["isbn"].isin(top_isbns)][["isbn","title","author"]].copy()
    result["hybrid_score"]  = result["isbn"].map(hybrid_scores)
    result["svd_score"]     = result["isbn"].map(svd_scores)
    result["content_score"] = result["isbn"].map(content_scores)
    result["graph_score"]   = result["isbn"].map(graph_scores)
    result = result.fillna(0).sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    return result


# ── UI ──────────────────────────────────────────────────────────
st.title("📚 Hybrid Book Recommendation System")
st.markdown("Combining **Collaborative Filtering (SVD)** + **Content-Based (TF-IDF)** + **Graph Search (A*)** for personalised recommendations.")

df, books_df, svd_data, tfidf_matrix, title_to_idx, G = load_models()

st.sidebar.header("Settings")

# User selection
all_users   = sorted(df["user_id"].unique().tolist())
user_id     = st.sidebar.selectbox("Select User ID", all_users[:200])

# Book title input
all_titles  = sorted(books_df["title"].dropna().unique().tolist())
liked_book  = st.sidebar.selectbox("Select a book you like", all_titles)

# Weights
st.sidebar.markdown("### Model Weights")
alpha = st.sidebar.slider("SVD (Collaborative)", 0.0, 1.0, 0.4, 0.05)
beta  = st.sidebar.slider("TF-IDF (Content)", 0.0, 1.0, 0.4, 0.05)
gamma = st.sidebar.slider("A* (Graph Search)", 0.0, 1.0, 0.2, 0.05)
n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

if st.sidebar.button("Get Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):
        recs = hybrid_recommend(
            user_id, liked_book, df, books_df,
            svd_data, tfidf_matrix, title_to_idx, G,
            alpha=alpha, beta=beta, gamma=gamma, n=n_recs
        )

    st.success(f"Top {n_recs} recommendations for User {user_id}")

    # Show tabs for hybrid vs individual models
    tab1, tab2, tab3, tab4 = st.tabs(["Hybrid Results", "SVD Only", "TF-IDF Only", "A* Only"])

    with tab1:
        st.subheader("Hybrid Recommendations")
        for i, row in recs.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.markdown(f"**{row['title']}**")
                col2.markdown(f"*{row['author']}*")
                col3.markdown(f"Score: `{row['hybrid_score']:.3f}`")
            st.divider()

    with tab2:
        st.subheader("SVD (Collaborative Filtering) Scores")
        svd_df = recs[["title","author","svd_score"]].sort_values("svd_score", ascending=False)
        st.dataframe(svd_df, use_container_width=True)

    with tab3:
        st.subheader("TF-IDF (Content-Based) Scores")
        content_df = recs[["title","author","content_score"]].sort_values("content_score", ascending=False)
        st.dataframe(content_df, use_container_width=True)

    with tab4:
        st.subheader("A* (Graph Search) Scores")
        graph_df = recs[["title","author","graph_score"]].sort_values("graph_score", ascending=False)
        st.dataframe(graph_df, use_container_width=True)

    # Score comparison chart
    st.subheader("Score Comparison Across Models")
    chart_df = recs[["title","svd_score","content_score","graph_score","hybrid_score"]].set_index("title")
    st.bar_chart(chart_df)

else:
    st.info("Select a user and a book you like from the sidebar, then click **Get Recommendations**.")
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    col1.metric("SVD Model", "Collaborative", "Learns user preferences from ratings")
    col2.metric("TF-IDF Model", "Content-Based", "Matches books by description similarity")
    col3.metric("A* Search", "Graph-Based", "Explores book similarity graph for diversity")
