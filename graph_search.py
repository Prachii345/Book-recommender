import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────
print("Loading data for graph search...")
books = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))

with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb") as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(MODEL_DIR, "isbn_to_idx.pkl"), "rb") as f:
    isbn_to_idx = pickle.load(f)

with open(os.path.join(MODEL_DIR, "title_to_idx.pkl"), "rb") as f:
    title_to_idx = pickle.load(f)

# ── Build book similarity graph ─────────────────────────────────
# We use a SAMPLE of books (first 3000) to keep graph small
print("Building book similarity graph (this may take ~30 seconds)...")

SAMPLE_SIZE = 3000
books_sample = books.iloc[:SAMPLE_SIZE].copy()
matrix_sample = tfidf_matrix[:SAMPLE_SIZE]

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(matrix_sample)

# Build graph: edge if similarity > threshold
THRESHOLD = 0.3
G = nx.Graph()

# Add nodes
for i, row in books_sample.iterrows():
    G.add_node(row["isbn"], title=row["title"], author=row["author"])

# Add edges
isbns = books_sample["isbn"].tolist()
for i in range(SAMPLE_SIZE):
    for j in range(i + 1, SAMPLE_SIZE):
        if sim_matrix[i][j] > THRESHOLD:
            G.add_edge(isbns[i], isbns[j], weight=sim_matrix[i][j])

print(f"  Graph nodes : {G.number_of_nodes()}")
print(f"  Graph edges : {G.number_of_edges()}")

# Save graph
with open(os.path.join(MODEL_DIR, "book_graph.pkl"), "wb") as f:
    pickle.dump(G, f)

print("Saved: models/book_graph.pkl")


# ── A* heuristic search for diverse recommendations ─────────────
def heuristic(node, goal, G):
    """Heuristic: 1 - edge weight if edge exists, else 1."""
    if G.has_edge(node, goal):
        return 1 - G[node][goal]["weight"]
    return 1.0


def astar_recommendations(start_isbn, G, books, n=10):
    """
    Use A* to explore the book graph from start_isbn and
    return n diverse but related books.
    """
    if start_isbn not in G:
        print(f"  ISBN {start_isbn} not in graph.")
        return pd.DataFrame()

    visited   = set()
    result    = []
    frontier  = [start_isbn]

    while frontier and len(result) < n:
        # Pick node with highest average edge weight (most connected)
        frontier.sort(
            key=lambda x: np.mean([G[x][nb]["weight"]
                                   for nb in G.neighbors(x)] or [0]),
            reverse=True
        )
        current = frontier.pop(0)

        if current in visited or current == start_isbn:
            visited.add(current)
            # Add unvisited neighbours to frontier
            for nb in G.neighbors(current):
                if nb not in visited:
                    frontier.append(nb)
            continue

        visited.add(current)
        result.append(current)

        for nb in G.neighbors(current):
            if nb not in visited:
                frontier.append(nb)

    result_books = books[books["isbn"].isin(result)][["isbn", "title", "author"]].copy()
    return result_books.reset_index(drop=True)


if __name__ == "__main__":
    sample_isbn = books_sample["isbn"].iloc[0]
    sample_title = books_sample["title"].iloc[0]
    print(f"\nA* recommendations starting from: '{sample_title}'")
    recs = astar_recommendations(sample_isbn, G, books, n=5)
    if not recs.empty:
        print(recs[["title", "author"]].to_string(index=False))
    print("\nGraph search done!")
