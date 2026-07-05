from jiwer import wer

def validate(reference, hypothesis):
    return wer(reference, hypothesis)
