import json
import os
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from src.parsers.log_parser import SessionFeatureExtractor
from src.models.hybrid_classifier import HybridBotDetector

detector = HybridBotDetector(web_log_weight=0.5, cnn_weight=0.5, cnn_high_confidence_thresh=0.85)


def score_web_logs(extractor: SessionFeatureExtractor, session_id: str) -> float:
    feat = extractor.session_features.get(session_id)
    if not feat or feat["Total_requests"] == 0:
        return 0.12

    tot = float(feat["Total_requests"])
    post_ratio = feat["Total_POST_requests"] / tot
    img_ratio = feat["image_requests"] / tot
    speed = feat["Browsing_speed"]

    prob = 0.15
    if post_ratio > 0.6:
        prob += 0.35
    if img_ratio < 0.05 and tot > 2:
        prob += 0.25
    if speed > 2.0:
        prob += 0.35
    if feat["Depth_SD"] == 0.0 and tot > 3:
        prob += 0.15

    return float(min(0.99, max(0.01, prob)))


def score_mouse_points(points: list) -> float:
    if not points or len(points) < 3:
        return 0.15

    pts = np.array(points)
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    distances = np.sqrt(dx**2 + dy**2)

    std_dist = np.std(distances)
    angles = np.arctan2(dy, dx)
    angle_diffs = np.abs(np.diff(angles))

    mean_angle_change = np.mean(angle_diffs) if len(angle_diffs) > 0 else 0.0

    prob = 0.18
    if std_dist < 1.5:
        prob += 0.42
    if mean_angle_change < 0.05:
        prob += 0.35

    return float(min(0.99, max(0.01, prob)))


class BotDetectionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy logging
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            index_path = os.path.join("static", "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/v1/evaluate":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                payload = {"logs": [], "mouse_points": []}

            logs = payload.get("logs", [])
            mouse_points = payload.get("mouse_points", [])

            extractor = SessionFeatureExtractor()
            session_id = "live_sess"
            for line in logs:
                extractor.add_log(line)

            prob_web = score_web_logs(extractor, session_id)
            prob_mouse = score_mouse_points(mouse_points)
            prob_hybrid = detector.combine_probabilities(prob_web, prob_mouse)

            if prob_hybrid >= 0.85:
                action = "BLOCK"
            elif prob_hybrid >= 0.5:
                action = "CHALLENGE"
            else:
                action = "ALLOW"

            feat = extractor.session_features.get(session_id, {})
            tot = max(1, feat.get("Total_requests", 1))

            res = {
                "action": action,
                "prob_web": prob_web,
                "prob_mouse": prob_mouse,
                "prob_hybrid": prob_hybrid,
                "is_bot": prob_hybrid >= 0.5,
                "features": {
                    "Total_requests": feat.get("Total_requests", 0),
                    "Browsing_speed": feat.get("Browsing_speed", 0.0),
                    "image_ratio": feat.get("image_requests", 0) / tot,
                    "Max_sequential_request": feat.get("Max_sequential_request", 0)
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_server(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, BotDetectionHandler)
    print(f"[ONLINE] Web Bot Detection Shield Server ready at http://127.0.0.1:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server(8000)
