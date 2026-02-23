from sklearn.metrics import roc_auc_score


def compute_roc_auc_ovr(y_true_bin, y_prob):
    return roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
