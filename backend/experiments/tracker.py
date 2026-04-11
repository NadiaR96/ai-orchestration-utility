import json
import uuid
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    def __init__(self, log_file: str = "experiments/logs.jsonl"):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: dict):
        record = {
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record