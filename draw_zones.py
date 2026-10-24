"""Draw one polygon per table on a camera frame and save them to tables.json.

Usage:
    python draw_zones.py rtsp://user:pass@192.168.1.20/stream1
    python draw_zones.py photo_of_the_room.jpg

Mouse : left click = add a corner of the current table
Keys  : n = finish this table, start the next (T1, T2, ...)
        u = undo last corner      s = save and quit      q = quit without saving
"""
import json
import sys

import cv2
import numpy as np


def grab_frame(src):
    if src.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(src)
    else:
        cap = cv2.VideoCapture(src)
        ok, img = cap.read()
        cap.release()
        if not ok:
            img = None
    if img is None:
        sys.exit("Could not read an image from: " + src)
    return img


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = grab_frame(sys.argv[1])
    tables, current = {}, []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])

    cv2.namedWindow("zones")
    cv2.setMouseCallback("zones", on_mouse)

    while True:
        view = base.copy()
        for name, pts in tables.items():
            p = np.array(pts, np.int32)
            cv2.polylines(view, [p], True, (0, 200, 0), 2)
            cv2.putText(view, name, tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if current:
            cv2.polylines(view, [np.array(current, np.int32)], False, (0, 0, 255), 2)
        label = f"Drawing T{len(tables) + 1}  |  n=next table  u=undo  s=save  q=quit"
        cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("zones", view)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("n") and len(current) >= 3:
            tables[f"T{len(tables) + 1}"] = current.copy()
            current.clear()
        elif k == ord("u") and current:
            current.pop()
        elif k == ord("s"):
            if len(current) >= 3:
                tables[f"T{len(tables) + 1}"] = current.copy()
            with open("tables.json", "w") as f:
                json.dump(tables, f, indent=2)
            print(f"Saved {len(tables)} tables to tables.json")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
