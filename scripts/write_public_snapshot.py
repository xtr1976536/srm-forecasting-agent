from __future__ import annotations

import argparse
import json
from pathlib import Path

from snapshot import anonymized_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, default=Path("public_snapshots"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    snapshot = anonymized_snapshot(result)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"{snapshot['date']}_{snapshot['run_id']}.json"
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
