from .semantic import semantic_score, semantic_score_advanced
from .syntax import syntax_score

# BASIC VERSION (used in API)
def final_score(ref, hyp):
    """Basic scoring: Semantic + Syntax (NO grammar dependency)"""
    
    alpha = 0.6   # semantic weight
    beta = 0.4    # syntax weight

    sem = semantic_score(ref, hyp)
    syn = syntax_score(ref)

    final = (alpha * sem) + (beta * syn)

    return {
        "semantic": sem,
        "syntax": syn,
        "grammar": 0.8,
        "final": round(final, 4)
    }


# ADVANCED VERSION
def final_score_v2(ref, hyp):
    """Advanced scoring: Semantic + Syntax + dummy Grammar"""

    alpha = 0.4
    beta = 0.3
    gamma = 0.3

    sem = semantic_score_advanced(ref, hyp)
    syn = syntax_score(ref)

    grammar = 0.8  # dummy value (no Java dependency)

    final = (alpha * sem) + (beta * syn) + (gamma * grammar)

    return {
        "semantic": sem,
        "syntax": syn,
        "grammar": grammar,
        "final": round(final, 4)
    }


# TEST BLOCK
if __name__ == "__main__":
    ref = "The quick brown fox jumps over the lazy dog"
    hyp = "A fast brown fox leaps over a lazy dog"

    print("Basic Score:", final_score(ref, hyp))
    print("Advanced Score:", final_score_v2(ref, hyp))