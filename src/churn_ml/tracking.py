from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunTracker:
    def __init__(self, tracking_path: Path) -> None:
        self.tracking_path = tracking_path
        self.tracking_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        with self.tracking_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
