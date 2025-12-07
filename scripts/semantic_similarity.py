# scripts/semantic_similarity.py
from sentence_transformers import SentenceTransformer, util

# Load once at import time (fast to reuse). Model is reasonably small and fast.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

def get_semantic_similarity(text1: str, text2: str) -> float:
    """
    Return cosine similarity between two texts using sentence-transformers.
    Outputs a float between -1 and 1 (we usually see 0..1 for similarity of different texts).
    """
    if not text1:
        text1 = ""
    if not text2:
        text2 = ""

    emb1 = _model.encode(text1, convert_to_tensor=True)
    emb2 = _model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    # clamp to float
    return float(score)
