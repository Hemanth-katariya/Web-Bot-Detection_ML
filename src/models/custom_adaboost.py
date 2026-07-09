import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class DecisionStump:
    """
    A simple decision stump: splits on 1 feature with threshold and polarity.
    Predicts labels in {-1, +1}.
    """
    def __init__(self, feature_index=None, threshold=None, polarity=1):
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        f = X[:, self.feature_index]
        preds = np.ones(f.shape[0], dtype=int)
        if self.polarity == 1:
            preds[f > self.threshold] = -1
        else:
            preds[f <= self.threshold] = -1
        return preds


class CustomAdaBoost(BaseEstimator, ClassifierMixin):
    """
    AdaBoost from scratch using decision stumps.
    Supports binary classification with {0, 1} or {-1, 1} labels.
    """
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.stumps = []
        self.alphas = []
        self.feature_importances_ = None
        self.train_scores_ = []
        self.val_scores_ = []
        if random_state is not None:
            np.random.seed(random_state)

    def _prepare_inputs(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.ravel()
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        unique = np.unique(y)
        if set(unique) <= {0, 1}:
            y_trans = np.where(y == 0, -1, 1)
        elif set(unique) <= {-1, 1}:
            y_trans = y
        else:
            mapping = {unique[0]: -1, unique[1]: 1}
            y_trans = np.vectorize(mapping.get)(y)
        return X, y_trans

    def _best_stump(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        best_stump = None
        best_error = float("inf")
        best_pred = None

        for feature_i in range(n_features):
            values = X[:, feature_i]
            sorted_idx = np.argsort(values)
            sorted_values = values[sorted_idx]
            uniques = np.unique(sorted_values)
            if uniques.size == 1:
                thresholds = [uniques[0]]
            else:
                thresholds = (uniques[:-1] + uniques[1:]) / 2.0

            for thr in thresholds:
                for polarity in [1, -1]:
                    preds = np.ones(n_samples, dtype=int)
                    if polarity == 1:
                        preds[values > thr] = -1
                    else:
                        preds[values <= thr] = -1
                    mis = (preds != y).astype(float)
                    err = np.dot(sample_weights, mis)
                    if err < best_error:
                        best_error = err
                        best_stump = DecisionStump(feature_index=feature_i, threshold=thr, polarity=polarity)
                        best_pred = preds.copy()
        return best_stump, best_error, best_pred

    def fit(self, X, y, **fit_params):
        X_val = fit_params.get("X_val", None)
        y_val = fit_params.get("y_val", None)
        verbose = fit_params.get("verbose", False)

        X, y_trans = self._prepare_inputs(X, y)
        if X_val is not None and y_val is not None:
            X_val, y_val_trans = self._prepare_inputs(X_val, y_val)

        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples

        self.stumps = []
        self.alphas = []
        self.train_scores_ = []
        self.val_scores_ = []

        for _ in range(self.n_estimators):
            stump, err, pred = self._best_stump(X, y_trans, w)
            eps = 1e-10
            err = max(eps, min(err, 1 - eps))
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)

            w = w * np.exp(-alpha * y_trans * pred)
            w = w / np.sum(w)

            self.stumps.append(stump)
            self.alphas.append(alpha)

            y_scores = self.decision_function(X)
            y_pred = np.where(y_scores >= 0, 1, -1)
            self.train_scores_.append(np.mean(y_pred == y_trans))

            if X_val is not None and y_val is not None:
                val_scores = self.decision_function(X_val)
                val_pred = np.where(val_scores >= 0, 1, -1)
                self.val_scores_.append(np.mean(val_pred == y_val_trans))

            if err <= eps:
                if verbose:
                    print("Stopping early: perfect stump found.")
                break

        n_features = X.shape[1]
        fi = np.zeros(n_features)
        for stump, a in zip(self.stumps, self.alphas):
            fi[stump.feature_index] += abs(a)
        if fi.sum() > 0:
            fi = fi / fi.sum()
        self.feature_importances_ = fi
        return self

    def decision_function(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X = np.asarray(X)
        n_samples = X.shape[0]
        agg = np.zeros(n_samples, dtype=float)
        for stump, alpha in zip(self.stumps, self.alphas):
            preds = stump.predict(X)
            agg += alpha * preds
        return agg

    def predict(self, X):
        scores = self.decision_function(X)
        preds_signed = np.where(scores >= 0, 1, -1)
        return np.where(preds_signed == -1, 0, 1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs = 1 / (1 + np.exp(-2 * scores))
        return np.vstack([1 - probs, probs]).T

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
