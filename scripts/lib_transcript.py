"""Transcript normalization helpers for benchmark runtimes."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _parse_tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, list):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"input": arguments}
    if arguments is None:
        return {}
    return {"input": arguments}


def canonicalize_openclaw_transcript(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize OpenClaw transcript events to the benchmark canonical schema."""
    normalized: list[dict[str, Any]] = []

    for event in transcript:
        if event.get("type") != "message":
            normalized.append(event)
            continue

        message = dict(event.get("message", {}))
        role = message.get("role")
        content = message.get("content", [])

        if role == "assistant" and isinstance(content, list):
            new_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append({"type": "text", "text": str(item)})
                    continue

                if item.get("type") == "toolCall":
                    parsed_args = _parse_tool_arguments(item.get("arguments", item.get("params", {})))
                    tool_item = dict(item)
                    tool_item["arguments"] = parsed_args
                    tool_item["params"] = parsed_args
                    new_content.append(tool_item)
                    continue

                if item.get("type") == "text":
                    text_item = dict(item)
                    text_item["text"] = _coerce_text(item.get("text", ""))
                    new_content.append(text_item)
                    continue

                new_content.append(item)
            message["content"] = new_content
        elif role in {"user", "toolResult"} and not isinstance(content, list):
            message["content"] = [_coerce_text(content)]

        normalized.append({"type": "message", "message": message})

    return normalized


def nanobot_messages_to_canonical(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert nanobot session messages to the benchmark canonical transcript schema."""
    transcript: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "user":
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [_coerce_text(message.get("content", ""))],
                    },
                }
            )
            continue

        if role == "assistant":
            content_items: list[dict[str, Any]] = []
            text = _coerce_text(message.get("content", ""))
            if text:
                content_items.append({"type": "text", "text": text})

            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                parsed_args = _parse_tool_arguments(function.get("arguments", {}))
                content_items.append(
                    {
                        "type": "toolCall",
                        "id": tool_call.get("id"),
                        "name": function.get("name") or tool_call.get("name") or "",
                        "arguments": parsed_args,
                        "params": parsed_args,
                    }
                )

            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": content_items,
                    },
                }
            )
            continue

        if role == "tool":
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "content": [
                            {
                                "tool_call_id": message.get("tool_call_id"),
                                "name": message.get("name"),
                                "content": _coerce_text(message.get("content", "")),
                            }
                        ],
                    },
                }
            )
            continue

        transcript.append(
            {
                "type": "message",
                "message": {
                    "role": str(role or "unknown"),
                    "content": [_coerce_text(message.get("content", ""))],
                },
            }
        )

    return transcript