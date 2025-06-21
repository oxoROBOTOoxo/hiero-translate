"""
Scan data/processed/images/*/*/*.jpg
Write data/processed/train_val.csv   (filepath,label,split)
"""

import csv
import random
from pathlib import Path

random.seed(42)

root = Path("data/processed/images")
jpgs = list(root.rglob("*.jpg"))

rows = []
for p in jpgs:
    # .../images/<ClassFolder>/<filename>.jpg
    label = p.parent.name
    rows.append((p.as_posix(), label))

random.shuffle(rows)
cut = int(len(rows) * 0.9)  # 90 % train, 10 % val

csv_path = Path("data/processed/train_val.csv")
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filepath", "label", "split"])
    for i, (fp, label) in enumerate(rows):
        split = "train" if i < cut else "val"
        w.writerow([fp, label, split])

print(f"Wrote {csv_path} with {len(rows)} rows " f"({cut} train / {len(rows)-cut} val)")
