import pandas as pd
import numpy as np
import pickle
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load clean ratings ──────────────────────────────────────────
print("Loading ratings...")
df = pd.read_csv(os.path.join(DATA_DIR, "ratings_clean.csv"))
print(f"  Ratings shape: {df.shape}")

# ── Build user-item matrix ──────────────────────────────────────
print("Building user-item matrix...")

# Encode user_id and isbn as integer indices
df["user_idx"] = pd.Categorical(df["user_id"]).codes
df["book_idx"] = pd.Categorical(df["isbn"]).codes

n_users = df["user_idx"].nunique()
n_books = df["book_idx"].nunique()
print(f"  Users: {n_users}, Books: {n_books}")

# Build sparse matrix
sparse_matrix = csr_matrix(
    (df["rating"].values, (df["user_idx"].values, df["book_idx"].values)),
    shape=(n_users, n_books)
)

# ── Train SVD ───────────────────────────────────────────────────
print("Training SVD (TruncatedSVD with 50 factors)...")
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(sparse_matrix)
item_factors = svd.components_

print(f"  User factors shape: {user_factors.shape}")
print(f"  Item factors shape: {item_factors.shape}")

# ── Evaluate on a sample ────────────────────────────────────────
print("Evaluating...")
sample = df.sample(min(5000, len(df)), random_state=42)
u_idx = sample["user_idx"].values
b_idx = sample["book_idx"].values

predicted = np.array([
    np.dot(user_factors[u], item_factors[:, b])
    for u, b in zip(u_idx, b_idx)
])
predicted = np.clip(predicted, 1, 10)
actual = sample["rating"].values

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae  = mean_absolute_error(actual, predicted)
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

# ── Save everything ─────────────────────────────────────────────
user_enc = dict(zip(pd.Categorical(df["user_id"]).categories, range(n_users)))
book_enc = dict(zip(pd.Categorical(df["isbn"]).categories, range(n_books)))
book_dec = {v: k for k, v in book_enc.items()}

with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "wb") as f:
    pickle.dump({
        "svd": svd,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_enc": user_enc,
        "book_enc": book_enc,
        "book_dec": book_dec
    }, f)

print("Saved: models/svd_model.pkl")
print("Collaborative filtering done!")


# ── Helper: top-N recommendations for a user ───────────────────
def get_svd_recommendations(user_id, df, model_data, books_df, n=10):
    user_enc   = model_data["user_enc"]
    book_enc   = model_data["book_enc"]
    book_dec   = model_data["book_dec"]
    user_factors = model_data["user_factors"]
    item_factors = model_data["item_factors"]

    if user_id not in user_enc:
        print(f"User {user_id} not found.")
        return pd.DataFrame()

    u_idx  = user_enc[user_id]
    u_vec  = user_factors[u_idx]

    # Scores for all books
    scores = item_factors.T.dot(u_vec)

    # Exclude already rated books
    rated_isbns = set(df[df["user_id"] == user_id]["isbn"].tolist())
    rated_idxs  = {book_enc[i] for i in rated_isbns if i in book_enc}

    ranked = np.argsort(scores)[::-1]
    top_idxs = [i for i in ranked if i not in rated_idxs][:n]

    top_isbns = [book_dec[i] for i in top_idxs]
    result = books_df[books_df["isbn"].isin(top_isbns)][["isbn","title","author"]].copy()
    result["predicted_rating"] = result["isbn"].map(
        {book_dec[i]: round(float(scores[i]), 2) for i in top_idxs}
    )
    return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    books_df = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))
    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
        model_data = pickle.load(f)

    sample_user = df["user_id"].iloc[0]
    print(f"\nTop 5 recommendations for user {sample_user}:")
    recs = get_svd_recommendations(sample_user, df, model_data, books_df, n=5)
    if not recs.empty:
        print(recs[["title", "author", "predicted_rating"]].to_string(index=False))
