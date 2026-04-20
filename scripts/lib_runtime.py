"""Benchmark runtime adapters for OpenClaw and nanobot."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib_agent import (
    ModelValidationError,
    cleanup_agent_sessions,
    ensure_agent_exists,
    execute_openclaw_task,
    prepare_task_workspace,
    run_openclaw_prompt,
    slugify_model,
    validate_openrouter_model,
)
from lib_tasks import Task
from lib_transcript import canonicalize_openclaw_transcript, nanobot_messages_to_canonical


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeOptions:
    kind: str
    model_id: str
    skill_dir: Path
    run_id: str
    timeout_multiplier: float = 1.0
    verbose: bool = False
    nanobot_config: Path | None = None
    nanobot_path: Path | None = None
    # When set, fixtures are resolved from these roots (e.g. QwenClawBench data/.../assets).
    asset_search_dirs: Optional[List[Path]] = None


class BenchmarkRuntime(ABC):
    """Base runtime adapter used by the benchmark harness."""

    name: str

    def __init__(self, options: RuntimeOptions):
        self.options = options

    @property
    @abstractmethod
    def agent_id(self) -> str:
        raise NotImplementedError

    def validate_model(self) -> None:
        validate_openrouter_model(self.options.model_id)

    def prepare(self) -> None:
        """Prepare runtime state for this benchmark run."""

    @abstractmethod
    def execute_task(self, task: Task) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_judge_prompt(
        self,
        *,
        task_id: str,
        prompt: str,
        judge_model: str,
        judge_agent_prefix: str,
        judge_timeout_seconds: float,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_version(self) -> str | None:
        return None


class OpenClawRuntime(BenchmarkRuntime):
    name = "openclaw"

    def __init__(self, options: RuntimeOptions):
        super().__init__(options)
        model_slug = slugify_model(options.model_id)
        self._agent_id = f"bench-{model_slug}"
        self.agent_workspace = Path(f"/tmp/pinchbench/{options.run_id}/agent_workspace")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def prepare(self) -> None:
        ensure_agent_exists(self.agent_id, self.options.model_id, self.agent_workspace)
        cleanup_agent_sessions(self.agent_id)

    def execute_task(self, task: Task) -> Dict[str, Any]:
        result = execute_openclaw_task(
            task=task,
            agent_id=self.agent_id,
            model_id=self.options.model_id,
            run_id=self.options.run_id,
            timeout_multiplier=self.options.timeout_multiplier,
            skill_dir=self.options.skill_dir,
            verbose=self.options.verbose,
            asset_search_dirs=self.options.asset_search_dirs,
        )
        result["transcript"] = canonicalize_openclaw_transcript(result.get("transcript", []))
        result["runtime"] = self.name
        return result

    def run_judge_prompt(
        self,
        *,
        task_id: str,
        prompt: str,
        judge_model: str,
        judge_agent_prefix: str,
        judge_timeout_seconds: float,
    ) -> Dict[str, Any]:
        model_slug = slugify_model(judge_model)
        judge_agent_id = f"{judge_agent_prefix}-{model_slug}"
        judge_workspace = Path("/tmp/pinchbench/judge/workspace")
        ensure_agent_exists(judge_agent_id, judge_model, judge_workspace)

        result = run_openclaw_prompt(
            agent_id=judge_agent_id,
            prompt=prompt,
            workspace=Path(f"/tmp/pinchbench/judge/{task_id}"),
            timeout_seconds=judge_timeout_seconds,
        )
        result["transcript"] = canonicalize_openclaw_transcript(result.get("transcript", []))
        result["runtime"] = self.name
        return result

    def get_version(self) -> str | None:
        try:
            result = subprocess.run(
                ["openclaw", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None


class NanobotRuntime(BenchmarkRuntime):
    name = "nanobot"
    _MIN_VERSION = (3, 11)

    def __init__(self, options: RuntimeOptions):
        super().__init__(options)
        model_slug = slugify_model(options.model_id)
        self._agent_id = f"nanobot-{model_slug}"
        self.runtime_root = Path(f"/tmp/pinchbench/{options.run_id}/nanobot")
        self.runtime_root.mkdir(parents=True, exist_ok=True)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def validate_model(self) -> None:
        if sys.version_info < self._MIN_VERSION:
            version = ".".join(str(part) for part in self._MIN_VERSION)
            current = ".".join(str(part) for part in sys.version_info[:3])
            raise RuntimeError(
                f"nanobot runtime requires Python {version}+ (current interpreter: {current})"
            )
        validate_openrouter_model(self.options.model_id)

    def execute_task(self, task: Task) -> Dict[str, Any]:
        return asyncio.run(self._execute_task_async(task))

    def run_judge_prompt(
        self,
        *,
        task_id: str,
        prompt: str,
        judge_model: str,
        judge_agent_prefix: str,
        judge_timeout_seconds: float,
    ) -> Dict[str, Any]:
        return asyncio.run(
            self._run_judge_prompt_async(
                task_id=task_id,
                prompt=prompt,
                judge_model=judge_model,
                judge_agent_prefix=judge_agent_prefix,
                judge_timeout_seconds=judge_timeout_seconds,
            )
        )

    def get_version(self) -> str | None:
        self._prepare_import_path()
        try:
            from nanobot import __version__  # type: ignore
        except Exception:
            return None
        return str(__version__)

    def _prepare_import_path(self) -> None:
        candidates: list[Path] = []
        if self.options.nanobot_path is not None:
            candidates.append(self.options.nanobot_path.expanduser().resolve())
        sibling_repo = self.options.skill_dir.parent / "nanobot"
        if sibling_repo.exists():
            candidates.append(sibling_repo.resolve())

        for candidate in candidates:
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)

    def _load_nanobot_components(self) -> tuple[Any, Any]:
        self._prepare_import_path()
        try:
            from nanobot.config.loader import set_config_path  # type: ignore
            from nanobot.nanobot import Nanobot  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "nanobot runtime is not available. Install nanobot-ai or pass --nanobot-path."
            ) from exc
        return Nanobot, set_config_path

    def _build_runtime_config(self, model_id: str, destination: Path) -> Path:
        source = self.options.nanobot_config or (Path.home() / ".nanobot" / "config.json")
        data: dict[str, Any] = {}
        if source.exists():
            data = json.loads(source.read_text(encoding="utf-8"))
        else:
            logger.warning("nanobot config not found at %s; using default config skeleton", source)

        agents = data.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})

        configured_model = model_id
        configured_provider = "auto"
        if model_id.startswith("openrouter/"):
            configured_provider = "openrouter"
            configured_model = model_id.split("/", 1)[1]

        defaults["model"] = configured_model
        defaults["provider"] = configured_provider

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return destination

    @staticmethod
    def _empty_usage() -> dict[str, Any]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }

    @staticmethod
    def _merge_usage(target: dict[str, Any], addition: dict[str, Any]) -> None:
        for key, value in addition.items():
            if key == "cost_usd":
                target[key] = float(target.get(key, 0.0)) + float(value or 0.0)
            else:
                target[key] = int(target.get(key, 0)) + int(value or 0)

    @staticmethod
    def _map_nanobot_usage(usage: dict[str, Any], request_count: int) -> dict[str, Any]:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        cached_tokens = int(usage.get("cached_tokens", 0) or 0)
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cache_read_tokens": cached_tokens,
            "cache_write_tokens": 0,
            "total_tokens": total_tokens,
            "cost_usd": 0.0,
            "request_count": request_count,
        }

    @staticmethod
    def _extract_session_prompt(session_entry: Any) -> str:
        if isinstance(session_entry, str):
            return session_entry
        if isinstance(session_entry, dict):
            return session_entry.get("prompt") or session_entry.get("message", "")
        return ""

    async def _execute_task_async(self, task: Task) -> Dict[str, Any]:
        Nanobot, set_config_path = self._load_nanobot_components()
        runtime_config = self._build_runtime_config(
            self.options.model_id,
            self.runtime_root / "config.json",
        )
        workspace = prepare_task_workspace(
            self.options.skill_dir,
            self.options.run_id,
            task,
            self.agent_id,
            workspace=Path(f"/tmp/pinchbench/{self.options.run_id}/{task.task_id}"),
            installed_skills_dirs=[Path.home() / ".nanobot" / "workspace" / "skills"],
            asset_search_dirs=self.options.asset_search_dirs,
        )

        start_time = time.time()
        timeout_seconds = task.timeout_seconds * self.options.timeout_multiplier
        transcript: list[dict[str, Any]] = []
        usage = self._empty_usage()
        stderr = ""
        status = "success"
        timed_out = False
        exit_code = 0
        bot = None

        sessions = task.frontmatter.get("sessions") or [{"prompt": task.prompt}]
        shared_session_key = f"bench:{task.task_id}"

        try:
            set_config_path(runtime_config)
            bot = Nanobot.from_config(config_path=runtime_config, workspace=workspace)
            current_session_key = shared_session_key

            for index, session_entry in enumerate(sessions, 1):
                prompt = self._extract_session_prompt(session_entry)
                if not prompt:
                    continue

                if isinstance(session_entry, dict) and session_entry.get("new_session"):
                    current_session_key = f"{shared_session_key}:{index}"

                session = bot._loop.sessions.get_or_create(current_session_key)
                before_count = len(session.messages)

                elapsed = time.time() - start_time
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    timed_out = True
                    status = "timeout"
                    break

                try:
                    await asyncio.wait_for(
                        bot._loop.process_direct(
                            prompt,
                            session_key=current_session_key,
                            channel="cli",
                            chat_id="benchmark",
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    status = "timeout"
                    break

                session = bot._loop.sessions.get_or_create(current_session_key)
                new_messages = session.messages[before_count:]
                transcript.extend(nanobot_messages_to_canonical(new_messages))

                request_count = sum(1 for message in new_messages if message.get("role") == "assistant")
                self._merge_usage(usage, self._map_nanobot_usage(bot._loop._last_usage, request_count))
        except Exception as exc:
            stderr = str(exc)
            status = "error"
            exit_code = -1
        finally:
            if bot is not None:
                await bot._loop.close_mcp()

        execution_time = time.time() - start_time
        if not transcript and status == "success":
            status = "error"

        return {
            "agent_id": self.agent_id,
            "task_id": task.task_id,
            "status": status,
            "transcript": transcript,
            "usage": usage,
            "workspace": str(workspace),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_time": execution_time,
            "stdout": "",
            "stderr": stderr,
            "runtime": self.name,
        }

    async def _run_judge_prompt_async(
        self,
        *,
        task_id: str,
        prompt: str,
        judge_model: str,
        judge_agent_prefix: str,
        judge_timeout_seconds: float,
    ) -> Dict[str, Any]:
        Nanobot, set_config_path = self._load_nanobot_components()
        judge_config = self._build_runtime_config(
            judge_model,
            self.runtime_root / f"judge-{slugify_model(judge_model)}.json",
        )
        workspace = Path(f"/tmp/pinchbench/judge/{task_id}")
        workspace.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        transcript: list[dict[str, Any]] = []
        usage = self._empty_usage()
        stderr = ""
        status = "success"
        timed_out = False
        exit_code = 0
        bot = None

        try:
            set_config_path(judge_config)
            bot = Nanobot.from_config(config_path=judge_config, workspace=workspace)
            session_key = f"judge:{judge_agent_prefix}:{task_id}"
            session = bot._loop.sessions.get_or_create(session_key)
            before_count = len(session.messages)

            try:
                await asyncio.wait_for(
                    bot._loop.process_direct(
                        prompt,
                        session_key=session_key,
                        channel="cli",
                        chat_id="judge",
                    ),
                    timeout=judge_timeout_seconds,
                )
            except asyncio.TimeoutError:
                timed_out = True
                status = "timeout"

            session = bot._loop.sessions.get_or_create(session_key)
            new_messages = session.messages[before_count:]
            transcript.extend(nanobot_messages_to_canonical(new_messages))
            request_count = sum(1 for message in new_messages if message.get("role") == "assistant")
            self._merge_usage(usage, self._map_nanobot_usage(bot._loop._last_usage, request_count))
        except Exception as exc:
            stderr = str(exc)
            status = "error"
            exit_code = -1
        finally:
            if bot is not None:
                await bot._loop.close_mcp()

        execution_time = time.time() - start_time
        if not transcript and status == "success":
            status = "error"

        return {
            "agent_id": f"{judge_agent_prefix}-{slugify_model(judge_model)}",
            "status": status,
            "transcript": transcript,
            "usage": usage,
            "workspace": str(workspace),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_time": execution_time,
            "stdout": "",
            "stderr": stderr,
            "runtime": self.name,
        }


def create_runtime(options: RuntimeOptions) -> BenchmarkRuntime:
    if options.kind == "openclaw":
        return OpenClawRuntime(options)
    if options.kind == "nanobot":
        return NanobotRuntime(options)
    raise ValueError(f"Unsupported runtime: {options.kind}")