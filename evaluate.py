import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = "data"
MODEL_DIR = "models"

print("=" * 55)
print("   HYBRID BOOK RECOMMENDATION SYSTEM — EVALUATION")
print("=" * 55)

# ── Load data and models ────────────────────────────────────────
df       = pd.read_csv(os.path.join(DATA_DIR, "ratings_clean.csv"))
books_df = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))

with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
    svd_data = pickle.load(f)

user_factors = svd_data["user_factors"]
item_factors = svd_data["item_factors"]
user_enc     = svd_data["user_enc"]
book_enc     = svd_data["book_enc"]
book_dec     = svd_data["book_dec"]

# ── 1. RMSE and MAE ─────────────────────────────────────────────
print("\n[1] Computing RMSE and MAE on test sample...")

sample = df.sample(min(8000, len(df)), random_state=42)
actuals, preds = [], []

for _, row in sample.iterrows():
    uid  = row["user_id"]
    isbn = row["isbn"]
    if uid in user_enc and isbn in book_enc:
        u = user_enc[uid]
        b = book_enc[isbn]
        pred = float(np.dot(user_factors[u], item_factors[:, b]))
        pred = np.clip(pred, 1, 10)
        actuals.append(row["rating"])
        preds.append(pred)

actuals = np.array(actuals)
preds   = np.array(preds)

rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
mae  = float(np.mean(np.abs(actuals - preds)))

print(f"   RMSE : {rmse:.4f}")
print(f"   MAE  : {mae:.4f}")

# ── 2. Precision@K and Recall@K ─────────────────────────────────
print("\n[2] Computing Precision@10 and Recall@10...")

K = 10
precisions, recalls = [], []

# Sample users who have enough ratings
eligible_users = []
for uid in df["user_id"].unique():
    user_df = df[df["user_id"] == uid]
    liked   = user_df[user_df["rating"] >= 7]
    if len(liked) >= 3 and uid in user_enc:
        eligible_users.append(uid)

# Use up to 200 eligible users
test_users = eligible_users[:200]
print(f"   Evaluating on {len(test_users)} eligible users...")

for user_id in test_users:
    user_df = df[df["user_id"] == user_id]

    # Split: 80% train, 20% test
    test_sample  = user_df.sample(frac=0.2, random_state=42)
    liked_test   = set(test_sample[test_sample["rating"] >= 7]["isbn"].tolist())

    if len(liked_test) == 0:
        continue

    # Get SVD scores excluding ALL rated books
    rated_isbns = set(user_df["isbn"].tolist())
    rated_idxs  = {book_enc[i] for i in rated_isbns if i in book_enc}

    u_idx  = user_enc[user_id]
    u_vec  = user_factors[u_idx]
    scores = item_factors.T.dot(u_vec)

    ranked   = np.argsort(scores)[::-1]
    top_idxs = [i for i in ranked if i not in rated_idxs][:K]
    rec_set  = {book_dec[i] for i in top_idxs}

    hits      = len(rec_set & liked_test)
    precision = hits / K
    recall    = hits / len(liked_test)

    precisions.append(precision)
    recalls.append(recall)

avg_precision = float(np.mean(precisions)) if precisions else 0.0
avg_recall    = float(np.mean(recalls))    if recalls    else 0.0

print(f"   Precision@10 : {avg_precision:.4f}")
print(f"   Recall@10    : {avg_recall:.4f}")

# ── 3. Summary Table ────────────────────────────────────────────
print("\n" + "=" * 55)
print("   FINAL EVALUATION SUMMARY")
print("=" * 55)
print(f"   {'Metric':<20} {'Model':<25} {'Value':>8}")
print("   " + "-" * 52)
print(f"   {'RMSE':<20} {'SVD Collaborative':<25} {rmse:>8.4f}")
print(f"   {'MAE':<20} {'SVD Collaborative':<25} {mae:>8.4f}")
print(f"   {'Precision@10':<20} {'SVD Collaborative':<25} {avg_precision:>8.4f}")
print(f"   {'Recall@10':<20} {'SVD Collaborative':<25} {avg_recall:>8.4f}")
print("=" * 55)
print("\nEvaluation complete!")
