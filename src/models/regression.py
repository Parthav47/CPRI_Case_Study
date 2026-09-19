"""Final reference-parameter regressor."""

from sklearn.ensemble import RandomForestRegressor


def create_regressor() -> RandomForestRegressor:
	return RandomForestRegressor(
		n_estimators=500, max_depth=None, min_samples_split=2,
		min_samples_leaf=1, max_features=1.0, random_state=42, n_jobs=-1,
	)
