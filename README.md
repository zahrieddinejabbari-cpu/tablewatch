# Table watcher

Camera -> YOLO -> table zones -> states (free / occupied / to clear) -> timers -> Telegram alerts -> SQLite stats.
Everything runs locally; only Telegram alerts use the internet.

## Setup
    pip install -r requirements.txt
    python draw_zones.py rtsp://user:pass@CAMERA_IP/stream1   # draw each table, press s
    export TELEGRAM_TOKEN=...  TELEGRAM_CHAT_ID=...           # optional
    python monitor.py rtsp://user:pass@CAMERA_IP/stream1 --show
    python report.py                                          # weekly stats

Test first with a video file or a webcam (`python monitor.py 0 --show`).

## Telegram
Talk to @BotFather -> /newbot -> copy the token. Send your bot a message, then open
https://api.telegram.org/bot<TOKEN>/getUpdates to read your chat id.

## Tuning (top of monitor.py)
OCCUPY_S, LEAVE_S, DISH_GONE_S, LONG_STAY_S, TO_CLEAR_ALERT_S. Expect 1-2 weeks of adjusting
zones and thresholds per room (angle, light, tables close together).

## Known limits
- COCO has no "plate" class: dishes = cups, glasses, bottles, bowls, cutlery. Fine-tune YOLO on
  your own photos if you want plates.
- No waiter detection yet. "Occupied for N min" is a plain timer. To alert on "no waiter visit",
  detect staff (fine-tune a class on your uniform, or color-filter the person crop) and reset a
  per-table `last_waiter_visit` when staff enter the zone.
- Overhead cameras see people from above; if detection is weak, try yolo11s.pt/yolo11m.pt.
- Privacy: no face recognition, no recording. Inform customers and staff (CNIL / CNDP rules).
