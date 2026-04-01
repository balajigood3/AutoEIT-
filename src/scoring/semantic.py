from sentence_transformers import SentenceTransformer, util, CrossEncoder
# Load models with distinct names to avoid overwriting
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

def semantic_score(ref, hyp):
    """Fast comparison using Cosine Similarity (Bi-Encoder)."""
    emb1 = bi_encoder.encode(ref, convert_to_tensor=True)
    emb2 = bi_encoder.encode(hyp, convert_to_tensor=True)
    
    score = util.cos_sim(emb1, emb2)
    return float(score.item()) # Use .item() to get a clean float from the tensor

def semantic_score_advanced(ref, hyp):
    """Slower but more accurate comparison (Cross-Encoder)."""
    score = cross_encoder.predict([(ref, hyp)])
    return float(score[0])