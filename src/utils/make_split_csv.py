"""
Scan data/processed/images/*/*/*.jpg
Create data/processed/train_val.csv  with columns:
filepath,label,label_id,split
"""

import csv
import random
from pathlib import Path

random.seed(42)

root = Path("data/processed/images")  # <- processed train images
jpgs = list(root.rglob("*.jpg"))

# ---------------------------------------------------------------------------
#   collect (filepath, label) pairs
# ---------------------------------------------------------------------------
rows = [(p.as_posix(), p.parent.name) for p in jpgs]

# ---------------------------------------------------------------------------
#   build a stable label→numeric-ID map  (e.g.  A1→0, D36→1 …)
# ---------------------------------------------------------------------------
labels = sorted({label for _, label in rows})
label2id = {lbl: idx for idx, lbl in enumerate(labels)}  # dict lookup

# ---------------------------------------------------------------------------
#   shuffle + split 90 / 10
# ---------------------------------------------------------------------------
random.shuffle(rows)
cut = int(len(rows) * 0.9)

# ---------------------------------------------------------------------------
#   write the CSV  (now with label_id column)
# ---------------------------------------------------------------------------
csv_path = Path("data/processed/train_val.csv")
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filepath", "label", "label_id", "split"])
    for i, (fp, label) in enumerate(rows):
        split = "train" if i < cut else "val"
        w.writerow([fp, label, label2id[label], split])

print(
    f"Wrote {csv_path} with {len(rows)} rows "
    f"({cut} train / {len(rows)-cut} val) and {len(label2id)} classes."
)
