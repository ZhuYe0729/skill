from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_transcript import canonicalize_openclaw_transcript, nanobot_messages_to_canonical  # noqa: E402


class TranscriptNormalizationTests(unittest.TestCase):
    def test_openclaw_tool_calls_gain_params_alias(self) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "name": "read_file",
                            "arguments": '{"path": "notes.md"}',
                        }
                    ],
                },
            }
        ]

        normalized = canonicalize_openclaw_transcript(transcript)

        item = normalized[0]["message"]["content"][0]
        self.assertEqual(item["arguments"], {"path": "notes.md"})
        self.assertEqual(item["params"], {"path": "notes.md"})

    def test_nanobot_messages_convert_to_canonical_schema(self) -> None:
        messages = [
            {"role": "user", "content": "Read notes.md"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "notes.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "hello",
            },
            {"role": "assistant", "content": "Done."},
        ]

        transcript = nanobot_messages_to_canonical(messages)

        self.assertEqual(transcript[0]["message"]["role"], "user")
        tool_call = transcript[1]["message"]["content"][0]
        self.assertEqual(tool_call["type"], "toolCall")
        self.assertEqual(tool_call["name"], "read_file")
        self.assertEqual(tool_call["arguments"], {"path": "notes.md"})
        self.assertEqual(tool_call["params"], {"path": "notes.md"})
        self.assertEqual(transcript[2]["message"]["role"], "toolResult")
        self.assertEqual(transcript[3]["message"]["content"][0]["text"], "Done.")


if __name__ == "__main__":
    unittest.main()