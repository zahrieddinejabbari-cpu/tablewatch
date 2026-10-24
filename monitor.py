"""Table watcher: camera -> YOLO -> table states, timers, Telegram alerts, stats.

Usage:
    export TELEGRAM_TOKEN=123:ABC   TELEGRAM_CHAT_ID=987654321   (optional)
    python monitor.py rtsp://user:pass@192.168.1.20/stream1 [--show]

Needs tables.json (create it with draw_zones.py).
"""
import argparse
import json
import os
import sqlite3
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ---------------------------------------------------------------- settings
MODEL = "yolo11n.pt"            # small & fast; try yolo11s.pt if detection is weak
PERSON = 0
DISHES = {39, 40, 41, 42, 43, 44, 45}  # bottle, wine glass, cup, fork, knife, spoon, bowl
CONF = 0.35
FRAME_STRIDE = 5                # analyse 1 frame out of N (RTSP is ~25 fps)

OCCUPY_S = 30                   # person must stay this long -> table "occupied"
LEAVE_S = 45                    # nobody seen for this long -> clients left
DISH_GONE_S = 30                # no dishes seen for this long -> table cleared
LONG_STAY_S = 5 * 60            # alert: occupied this long (adapt: no-waiter rule, see README)
TO_CLEAR_ALERT_S = 3 * 60       # alert: table waiting to be cleared this long
ALERT_COOLDOWN_S = 10 * 60

FREE, OCCUPIED, TO_CLEAR = "free", "occupied", "to_clear"
LABEL = {FREE: "libre", OCCUPIED: "occupée", TO_CLEAR: "à débarrasser"}

# ---------------------------------------------------------------- output
db = sqlite3.connect("stats.db")
db.execute("CREATE TABLE IF NOT EXISTS events (ts REAL, tbl TEXT, old TEXT, new TEXT, "
           "duration_s REAL)")
db.commit()


def send_alert(text):
    print("ALERT:", text, flush=True)
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=5)
    except requests.RequestException as e:
        print("Telegram error:", e)


# ---------------------------------------------------------------- logic
class Table:
    def __init__(self, name, polygon):
        self.name = name
        self.poly = np.array(polygon, np.int32).reshape(-1, 1, 2)
        self.state, self.since = FREE, time.time()
        self.first_person = self.last_person = self.last_dish = None
        self.last_alert = 0.0

    def inside(self, point):
        return cv2.pointPolygonTest(self.poly, (float(point[0]), float(point[1])), False) >= 0

    def set_state(self, new, now):
        dur = now - self.since
        db.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                   (now, self.name, self.state, new, dur))
        db.commit()
        print(f"{time.strftime('%H:%M:%S')} {self.name}: {self.state} -> {new} "
              f"(after {dur:.0f}s)", flush=True)
        self.state, self.since, self.last_alert = new, now, 0.0

    def update(self, person_here, dish_here, now):
        if person_here:
            self.last_person = now
            if self.first_person is None:
                self.first_person = now
        person_gone = self.last_person is None or now - self.last_person > LEAVE_S
        if person_gone:
            self.first_person = None
        if dish_here:
            self.last_dish = now
        dish_gone = self.last_dish is None or now - self.last_dish > DISH_GONE_S
        seated = self.first_person is not None and now - self.first_person >= OCCUPY_S

        if self.state == FREE and seated:
            self.set_state(OCCUPIED, now)
        elif self.state == OCCUPIED and person_gone:
            self.set_state(FREE if dish_gone else TO_CLEAR, now)
        elif self.state == TO_CLEAR:
            if seated:
                self.set_state(OCCUPIED, now)
            elif dish_gone:
                self.set_state(FREE, now)

        # alerts
        age = now - self.since
        if now - self.last_alert > ALERT_COOLDOWN_S:
            if self.state == OCCUPIED and age > LONG_STAY_S:
                send_alert(f"⏱ {self.name} occupée depuis {age // 60:.0f} min")
                self.last_alert = now
            elif self.state == TO_CLEAR and age > TO_CLEAR_ALERT_S:
                send_alert(f"🧹 {self.name} à débarrasser depuis {age // 60:.0f} min")
                self.last_alert = now


def write_status(tables, now):
    data = {t.name: {"state": t.state, "seconds": int(now - t.since)} for t in tables}
    with open("status.json", "w") as f:
        json.dump({"updated": now, "tables": data}, f)


def draw(frame, tables):
    colors = {FREE: (0, 200, 0), OCCUPIED: (0, 0, 230), TO_CLEAR: (0, 190, 230)}
    now = time.time()
    for t in tables:
        cv2.polylines(frame, [t.poly], True, colors[t.state], 2)
        x, y = t.poly[0][0]
        cv2.putText(frame, f"{t.name} {LABEL[t.state]} {int(now - t.since) // 60}m",
                    (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[t.state], 2)
    return frame


def run(source, show):
    with open("tables.json") as f:
        tables = [Table(n, p) for n, p in json.load(f).items()]
    model = YOLO(MODEL)
    last_status = 0.0
    for res in model.predict(source, stream=True, classes=[PERSON, *DISHES],
                             conf=CONF, vid_stride=FRAME_STRIDE, verbose=False):
        now = time.time()
        people, dishes = [], []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c in zip(xyxy, cls):
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                (people if c == PERSON else dishes).append(center)
        for t in tables:
            t.update(any(t.inside(p) for p in people),
                     any(t.inside(d) for d in dishes), now)
        if now - last_status > 5:
            write_status(tables, now)
            last_status = now
        if show:
            cv2.imshow("tables", draw(res.orig_img.copy(), tables))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="RTSP url, video file, or 0 for a webcam")
    ap.add_argument("--show", action="store_true", help="live preview window")
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    while True:  # reconnect if the stream drops
        try:
            run(src, a.show)
            break
        except Exception as e:
            print("Stream error, retrying in 10s:", e)
            time.sleep(10)
