from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union


def append_feedback(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
