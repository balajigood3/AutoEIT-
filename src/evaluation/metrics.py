from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu

def evaluate(ref, hyp):
    return {
        "WER": wer(ref, hyp),
        "BLEU": sentence_bleu([ref.split()], hyp.split())
    }