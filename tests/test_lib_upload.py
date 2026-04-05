from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_upload import _build_payload  # noqa: E402


class UploadPayloadTests(unittest.TestCase):
    def test_build_payload_includes_runtime_metadata(self) -> None:
        raw = {
            "model": "openrouter/anthropic/claude-sonnet-4",
            "benchmark_version": "deadbeef",
            "runtime": "nanobot",
            "runtime_version": "0.1.4.post6",
            "run_id": "0001",
            "timestamp": 0,
            "suite": "task_00_sanity",
            "tasks": [
                {
                    "task_id": "task_00_sanity",
                    "execution_time": 1.2,
                    "timed_out": False,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "request_count": 1,
                        "cost_usd": 0.0,
                    },
                    "grading": {
                        "mean": 1.0,
                        "runs": [
                            {
                                "grading_type": "automated",
                                "max_score": 1.0,
                                "breakdown": {"agent_responded": 1.0},
                                "notes": "",
                            }
                        ],
                    },
                    "frontmatter": {},
                }
            ],
        }

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            payload = _build_payload(path)

        self.assertEqual(payload["runtime"], "nanobot")
        self.assertEqual(payload["runtime_version"], "0.1.4.post6")
        self.assertIsNone(payload["openclaw_version"])
        self.assertEqual(payload["metadata"]["runtime"], "nanobot")


if __name__ == "__main__":
    unittest.main()