import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading datasets...")

books = pd.read_csv(
    os.path.join(DATA_DIR, "Books.csv"),
    sep=";", encoding="latin-1", on_bad_lines="skip", low_memory=False
)
ratings = pd.read_csv(
    os.path.join(DATA_DIR, "Ratings.csv"),
    sep=";", encoding="latin-1", on_bad_lines="skip"
)
users = pd.read_csv(
    os.path.join(DATA_DIR, "Users.csv"),
    sep=";", encoding="latin-1", on_bad_lines="skip"
)

print(f"  Books   : {books.shape}")
print(f"  Ratings : {ratings.shape}")
print(f"  Users   : {users.shape}")
print(f"\n  Books columns   : {books.columns.tolist()}")
print(f"  Ratings columns : {ratings.columns.tolist()}")

# Standardise column names
books.columns   = [c.strip() for c in books.columns]
ratings.columns = [c.strip() for c in ratings.columns]
users.columns   = [c.strip() for c in users.columns]

books = books.rename(columns={
    "ISBN": "isbn", "Title": "title", "Author": "author",
    "Year": "year", "Publisher": "publisher"
})
ratings = ratings.rename(columns={
    "User-ID": "user_id", "ISBN": "isbn", "Rating": "rating"
})

# Clean books
books = books[["isbn", "title", "author", "year", "publisher"]].copy()
books.dropna(subset=["title", "author"], inplace=True)
books["title"]     = books["title"].str.strip().str.strip('"')
books["author"]    = books["author"].str.strip()
books["year"]      = books["year"].fillna("Unknown")
books["publisher"] = books["publisher"].fillna("Unknown")
print(f"\nBooks after cleaning : {books.shape}")

# Clean ratings
ratings = ratings[ratings["rating"] > 0].copy()
ratings.dropna(inplace=True)
print(f"Ratings after cleaning : {ratings.shape}")

# Filter sparse users & books
user_counts = ratings["user_id"].value_counts()
book_counts = ratings["isbn"].value_counts()
ratings = ratings[
    ratings["user_id"].isin(user_counts[user_counts >= 20].index) &
    ratings["isbn"].isin(book_counts[book_counts >= 20].index)
].copy()

print(f"Ratings after sparsity filter : {ratings.shape}")
print(f"  Active users : {ratings['user_id'].nunique()}")
print(f"  Active books : {ratings['isbn'].nunique()}")

# Merge
df = ratings.merge(books, on="isbn", how="inner")
print(f"\nMerged dataframe : {df.shape}")

# Content text for TF-IDF
books["content"] = (
    books["title"].fillna("") + " " +
    books["author"].fillna("") + " " +
    books["publisher"].fillna("")
)

# Save
df.to_csv(os.path.join(DATA_DIR, "ratings_clean.csv"), index=False)
books.to_csv(os.path.join(DATA_DIR, "books_clean.csv"), index=False)

print("\nSaved: data/ratings_clean.csv and data/books_clean.csv")
print("Preprocessing complete!")
