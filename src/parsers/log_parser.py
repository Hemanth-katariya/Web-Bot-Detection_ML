import re
from datetime import datetime
import numpy as np


class ApacheLogParser:
    """Parses Apache web server access log lines."""

    LOG_RE = re.compile(r'''
        (?P<host>\S+)\s+                # host (or -)
        (?P<ident>\S+)\s+               # ident (or -)
        \[(?P<time>[^\]]+)\]\s+         # [timestamp]
        "(?P<method>\S+)\s+(?P<path>[^"]+?)\s+(?P<proto>[^"]+)"\s+  # "METHOD path PROTO"
        (?P<status>\d{3})\s+            # status
        (?P<size>\S+)\s+                # size in bytes or -
        "(?P<referer>[^"]*)"\s+         # "referer" (may be empty)
        (?P<token>\S+)\s+               # custom token (session id or cookie)
        "(?P<agent>[^"]+)"              # "user-agent"
    ''', re.VERBOSE)

    @classmethod
    def parse_line(cls, line: str):
        m = cls.LOG_RE.search(line)
        if not m:
            return None
        d = m.groupdict()

        try:
            dt = datetime.strptime(d['time'], "%d/%b/%Y:%H:%M:%S %z")
        except ValueError:
            return None

        size = None if d['size'] == '-' else int(d['size'])

        return {
            'timestamp': dt,
            'method': d['method'],
            'path': d['path'],
            'protocol': d['proto'],
            'status': int(d['status']),
            'size': size or 0,
            'referer': None if d['referer'] == '' else d['referer'],
            'session': d['token'],
            'agent': d['agent'],
        }


class SessionFeatureExtractor:
    """Aggregates log lines per session into 17 numerical behavioral features."""

    IMAGE_EXTENSIONS = ('.png', '.jpeg', '.jpg', '.webp', '.svg')

    def __init__(self):
        self.session_features = {}
        self.parser = ApacheLogParser()

    @classmethod
    def is_image(cls, path: str) -> bool:
        """Corrected image detection matching all common image extensions."""
        if not path:
            return False
        return path.lower().endswith(cls.IMAGE_EXTENSIONS)

    def init_features(self):
        return {
            "Total_requests": 0,
            "Total_Bytes": 0,
            "Total_GET_requests": 0,
            "Total_POST_requests": 0,
            "Total_3xx_responses": 0,
            "Total_4xx_responses": 0,
            "image_requests": 0,
            "css_file_request": 0,
            "js_requests": 0,
            "Depth_SD": 0.0,
            "Max_requests_per_page": 0,
            "Average_requests_per_page": 0.0,
            "Max_sequential_request": 0,
            "per_sequential_requests": 0.0,
            "Session_time": 0.0,
            "Browsing_speed": 0.0,
            "SD_inter_request_time": 0.0,
            "request_path": [],
            "requests_timestamps": [],
            "Total_requests_no_mv": 0,
        }

    def calculate_depth_sd(self, session_id: str) -> float:
        paths = self.session_features[session_id]["request_path"]
        if not paths:
            return 0.0
        depths = [path.count('/') for path in paths]
        return float(np.std(depths))

    def calculate_inter_request_time_sd(self, session_id: str) -> float:
        timestamps = self.session_features[session_id]["requests_timestamps"]
        if len(timestamps) <= 1:
            return 0.0
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
            time_diffs.append(diff)
        return float(np.std(time_diffs))

    def get_request_stats(self, session_id: str):
        paths = self.session_features[session_id]["request_path"]
        if not paths:
            return {
                "Max_requests_per_page": 0,
                "Average_requests_per_page": 0.0,
                "Max_sequential_request": 0,
                "cnt_consecutive_path": 0,
                "Browsing_speed": 0.0,
            }

        request_path_cnt = {}
        max_consecutive_len = 1
        cur_consecutive_len = 0
        cnt_consecutive_path = 0
        prev_path = ""

        for path in paths:
            request_path_cnt[path] = request_path_cnt.get(path, 0) + 1
            if path.startswith(prev_path) and prev_path:
                cur_consecutive_len += 1
                cnt_consecutive_path += 1
                max_consecutive_len = max(max_consecutive_len, cur_consecutive_len)
            else:
                cur_consecutive_len = 1
            prev_path = path

        total_req = sum(request_path_cnt.values())
        max_req = max(request_path_cnt.values(), default=0)
        tot_pages = len(request_path_cnt)
        session_time = self.session_features[session_id]["Session_time"]

        return {
            "Max_requests_per_page": max_req,
            "Average_requests_per_page": total_req / tot_pages if tot_pages > 0 else 0.0,
            "Max_sequential_request": max_consecutive_len,
            "cnt_consecutive_path": cnt_consecutive_path,
            "Browsing_speed": tot_pages / session_time if session_time > 0 else 0.0,
        }

    def add_log(self, raw_line: str, max_req: int = 10000):
        parsed = self.parser.parse_line(raw_line)
        if not parsed or parsed["session"] == '-':
            return

        session_id = parsed["session"]
        if session_id not in self.session_features:
            self.session_features[session_id] = self.init_features()
            self.session_features[session_id]["session_start"] = parsed["timestamp"]

        if not (parsed["method"] == 'POST' and parsed["path"] == '/storage/store_sess_total_mousemv_db.php'):
            self.session_features[session_id]["Total_requests_no_mv"] += 1

        if self.session_features[session_id]["Total_requests_no_mv"] > max_req:
            return

        feat = self.session_features[session_id]
        feat["Total_requests"] += 1
        feat["Total_Bytes"] += parsed["size"]
        feat["Total_GET_requests"] += 1 if parsed["method"] == "GET" else 0
        feat["Total_POST_requests"] += 1 if parsed["method"] == "POST" else 0
        feat["Total_3xx_responses"] += 1 if parsed["status"] // 100 == 3 else 0
        feat["Total_4xx_responses"] += 1 if parsed["status"] // 100 == 4 else 0
        feat["image_requests"] += 1 if self.is_image(parsed["path"]) else 0
        feat["css_file_request"] += 1 if parsed["path"].lower().endswith(".css") else 0
        feat["js_requests"] += 1 if parsed["path"].lower().endswith(".js") else 0

        feat["request_path"].append(parsed["path"])
        feat["requests_timestamps"].append(parsed["timestamp"])

        feat["Depth_SD"] = self.calculate_depth_sd(session_id)
        feat["Session_time"] = (parsed["timestamp"] - feat["session_start"]).total_seconds()
        feat["SD_inter_request_time"] = self.calculate_inter_request_time_sd(session_id)
        feat.update(self.get_request_stats(session_id))

    def get_session_features_as_csv(self, session_id: str) -> str:
        feat = self.session_features.get(session_id)
        if not feat or feat["Total_requests"] == 0:
            return ""

        tot = float(feat["Total_requests"])
        row = [
            session_id,
            feat["Total_requests"],
            feat["Total_Bytes"],
            feat["Total_GET_requests"],
            feat["Total_POST_requests"],
            feat["Total_3xx_responses"] / tot,
            feat["Total_4xx_responses"] / tot,
            feat["image_requests"] / tot,
            feat["css_file_request"] / tot,
            feat["js_requests"] / tot,
            feat["Depth_SD"],
            feat["Max_requests_per_page"],
            feat["Average_requests_per_page"],
            feat["Max_sequential_request"],
            feat.get("cnt_consecutive_path", 0) / tot,
            feat["Session_time"],
            feat["Browsing_speed"],
            feat["SD_inter_request_time"],
        ]
        return ",".join(map(str, row)) + "\n"
