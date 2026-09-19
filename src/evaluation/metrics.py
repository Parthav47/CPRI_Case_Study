"""Pure metric calculations with no project-specific results."""

import numpy as np
from sklearn.metrics import (
	accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
	mean_absolute_error, mean_squared_error, precision_score, r2_score,
	recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_pred, invalid_probability=None) -> dict:
	metrics = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
	}
	if invalid_probability is not None:
		metrics["roc_auc"] = float(roc_auc_score(y_true, invalid_probability))
	return metrics


def regression_metrics(y_true, y_pred) -> dict:
	return {
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
		"r2": float(r2_score(y_true, y_pred)),
	}
