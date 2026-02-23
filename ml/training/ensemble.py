"""Weighted soft-voting ensemble and optional stacking placeholder."""

def weighted_soft_voting(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must have same length")
    total = sum(weights)
    combined = [0.0] * len(predictions[0])
    for probs, w in zip(predictions, weights):
        for i, p in enumerate(probs):
            combined[i] += p * w
    return [p / total for p in combined]
