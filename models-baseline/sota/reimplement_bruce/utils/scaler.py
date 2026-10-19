import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureMinMaxScaler:
    """
    MinMax Scaler for input tabular features to range [0, 1].
    Handles NaNs, Infs, and zero variance columns cleanly.
    """
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.valid_cols = None

    def fit_transform(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Drop columns that are completely zero or constant to avoid division by zero
        std = np.std(X, axis=0)
        self.valid_cols = std > 1e-12
        if not np.any(self.valid_cols):
            self.valid_cols = np.ones(X.shape[1], dtype=bool)
        
        X_filtered = X[:, self.valid_cols]
        X_scaled = self.scaler.fit_transform(X_filtered)
        return X_scaled

    def transform(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_filtered = X[:, self.valid_cols]
        X_scaled = self.scaler.transform(X_filtered)
        return X_scaled

class TargetNormalizer:
    """
    Normalizes regression targets before neural network training
    and inverse transforms predictions to original units.
    """
    def __init__(self, method='standard'):
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

    def fit_transform(self, y):
        y_arr = np.asarray(y).reshape(-1, 1)
        return self.scaler.fit_transform(y_arr).ravel()

    def transform(self, y):
        y_arr = np.asarray(y).reshape(-1, 1)
        return self.scaler.transform(y_arr).ravel()

    def inverse_transform(self, y_scaled):
        y_arr = np.asarray(y_scaled).reshape(-1, 1)
        return self.scaler.inverse_transform(y_arr).ravel()
