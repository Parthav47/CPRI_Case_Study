"""Final validity classifier and threshold helpers."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

INVALID_THRESHOLD = 0.415


def create_classifier() -> RandomForestClassifier:
	return RandomForestClassifier(
		n_estimators=500, max_depth=20, max_features=0.5,
		min_samples_split=2, min_samples_leaf=1, class_weight="balanced",
		random_state=42, n_jobs=-1,
	)


def predict_invalid_probability(model: RandomForestClassifier, features) -> np.ndarray:
	invalid_index = list(model.classes_).index(1)
	return model.predict_proba(features)[:, invalid_index]


def predict_validity(model: RandomForestClassifier, features, threshold: float = INVALID_THRESHOLD) -> np.ndarray:
	return np.where(predict_invalid_probability(model, features) >= threshold, "Invalid", "Valid")
