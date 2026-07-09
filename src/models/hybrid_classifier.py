import numpy as np


class HybridBotDetector:
    """
    Robust Hybrid Bot Detector combining Tabular Web Log predictions
    with Sequential Mouse Movement CNN predictions.
    """

    def __init__(self, web_log_weight: float = 0.5, cnn_weight: float = 0.5, cnn_high_confidence_thresh: float = 0.85):
        self.web_log_weight = web_log_weight
        self.cnn_weight = cnn_weight
        self.cnn_high_confidence_thresh = cnn_high_confidence_thresh

    def combine_probabilities(self, web_log_prob_bot: float, cnn_prob_bot: float) -> float:
        """
        Combines probabilities from both modalities.
        If CNN detects a bot with extreme confidence (e.g., > 0.85),
        we elevate bot probability appropriately. Otherwise we perform calibrated weighted fusion.
        """
        if cnn_prob_bot >= self.cnn_high_confidence_thresh:
            return max(cnn_prob_bot, 0.5 * web_log_prob_bot + 0.5 * cnn_prob_bot)

        combined = (
            self.web_log_weight * web_log_prob_bot +
            self.cnn_weight * cnn_prob_bot
        ) / (self.web_log_weight + self.cnn_weight)
        return float(combined)

    def predict(self, web_log_prob_bot: float, cnn_prob_bot: float, threshold: float = 0.5) -> int:
        combined_prob = self.combine_probabilities(web_log_prob_bot, cnn_prob_bot)
        return 1 if combined_prob >= threshold else 0

    def predict_batch(self, web_log_probs: np.ndarray, cnn_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        preds = []
        for w_prob, c_prob in zip(web_log_probs, cnn_probs):
            preds.append(self.predict(float(w_prob), float(c_prob), threshold))
        return np.array(preds, dtype=int)
