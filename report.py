"""Weekly stats from stats.db: average stay per table and rotations per hour."""
import sqlite3
import time
from collections import Counter

db = sqlite3.connect("stats.db")
since = time.time() - 7 * 86400

print("Average stay (minutes) per table:")
for tbl, avg, n in db.execute(
    "SELECT tbl, AVG(duration_s)/60, COUNT(*) FROM events "
    "WHERE old='occupied' AND ts>? GROUP BY tbl ORDER BY tbl", (since,)):
    print(f"  {tbl}: {avg:.0f} min over {n} services")

print("\nNew seatings by hour of day:")
hours = Counter(int(time.strftime("%H", time.localtime(ts))) for (ts,) in db.execute(
    "SELECT ts FROM events WHERE new='occupied' AND ts>?", (since,)))
for h in range(24):
    if hours[h]:
        print(f"  {h:02d}h {'█' * hours[h]} {hours[h]}")
