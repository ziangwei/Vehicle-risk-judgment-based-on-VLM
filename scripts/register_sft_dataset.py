import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
info_p = ROOT / "data/dataset_info.json"
backup = info_p.with_suffix(".bak")

if info_p.exists():
    backup.write_text(info_p.read_text(encoding="utf-8"), encoding="utf-8")
else:
    info_p.parent.mkdir(parents=True, exist_ok=True)
    info_p.write_text("{}", encoding="utf-8")

info = json.loads(info_p.read_text(encoding="utf-8"))

info.update({
    "bdd_multiframe_sft_train": {
        "file_name": "bdd_multiframe_sft_train.jsonl",
        "columns": {"images":"images","prompt":"prompt","response":"response"}
    },
    "bdd_multiframe_sft_val": {
        "file_name": "bdd_multiframe_sft_val.jsonl",
        "columns": {"images":"images","prompt":"prompt","response":"response"}
    }
})

info_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print("dataset_info.json updated. Backup at:", backup)
