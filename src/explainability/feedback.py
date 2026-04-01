def generate_feedback(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()

    missing = list(set(ref_words) - set(hyp_words))
    extra = list(set(hyp_words) - set(ref_words))

    return {
        "missing_words": missing,
        "extra_words": extra,
        "comment": "Check missing and incorrect words"
    }