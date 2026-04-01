from jiwer import wer

def syntax_score(ref, hyp):
    error = wer(ref, hyp)
    return 1 - error