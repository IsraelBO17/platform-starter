# scripts/embeddings_demo.py
"""Find the support article closest in meaning to a customer's question."""
from sentence_transformers import SentenceTransformer

# A small, free embedding model (about 90 MB) that runs on a laptop CPU.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

articles = [
    "How to reverse a failed transfer",
    "Changing the phone number on your account",
    "Why your card was declined abroad",
    "Opening a savings account for a child",
]
question = "I sent money but it never arrived, can I get it back?"

article_vectors = model.encode(articles, normalize_embeddings=True)   # shape (4, 384)
question_vector = model.encode(question, normalize_embeddings=True)    # shape (384,)

# With normalised vectors, the dot product IS the cosine similarity (-1 to 1).
scores = article_vectors @ question_vector
for article, score in sorted(zip(articles, scores), key=lambda pair: -pair[1]):
    print(f"{score:.2f}  {article}")
