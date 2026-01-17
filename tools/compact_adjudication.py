# tools/compact_adjudication.py
import json
from pathlib import Path
from collections import defaultdict

def compact_dataset(jsonl_path):
    data = []
    if not Path(jsonl_path).exists():
        print("No dataset found.")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Group by (source_file, approx_bbox)
    # Since bboxes are floats, we round them to cluster clicks on the same box
    groups = defaultdict(list)
    
    for entry in data:
        # create a key based on file and rounded bbox (to identify "same box" edits)
        bbox = entry.get("bbox")
        if bbox:
            # Round to nearest 5 pixels to group "same box" edits
            key_box = tuple(round(c / 5) * 5 for c in bbox)
            key = (entry["source_file"], key_box)
        else:
            key = (entry["source_file"], "no_box")
            
        groups[key].append(entry)

    final_dataset = []
    for key, entries in groups.items():
        # Sort by timestamp, keep last
        entries.sort(key=lambda x: x["timestamp"])
        final_dataset.append(entries[-1])

    print(f"Raw events: {len(data)}")
    print(f"Unique Corrected Equations: {len(final_dataset)}")
    
    # Optional: Dump to a 'clean' file
    # with open("data/adjudicated/clean_dataset.jsonl", "w") as f:
    #    for entry in final_dataset:
    #        f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    compact_dataset("data/adjudicated/dataset.jsonl")