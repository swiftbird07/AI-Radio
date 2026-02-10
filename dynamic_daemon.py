#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hmac
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class DaemonSettings:
    repo_root: Path
    python_bin: str
    default_workdir: str
    default_output_dir: str
    dynamic_poll_seconds: int
    auth_token: str


@dataclass
class DaemonState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen[bytes] | None = None
    started_at_utc: str | None = None
    last_command: list[str] = field(default_factory=list)
    last_params: dict[str, Any] = field(default_factory=dict)
    last_exit_code: int | None = None


class DynamicDaemonHandler(BaseHTTPRequestHandler):
    server_version = "AIRadioDynamicDaemon/1.0"
    daemon_settings: DaemonSettings
    daemon_state: DaemonState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)

        if parsed.path == "/start-dynamic":
            self._handle_start_dynamic(query)
            return
        if parsed.path == "/stop-dynamic":
            self._handle_stop_dynamic(query)
            return
        if parsed.path == "/status":
            self._handle_status(query)
            return

        self._json_response(
            404,
            {
                "error": "not_found",
                "message": f"Unknown path '{parsed.path}'.",
            },
        )

    def _handle_start_dynamic(self, query: dict[str, list[str]]) -> None:
        if not self._check_auth(query):
            return

        playback_id = self._first_query(query, "playback-id") or self._first_query(
            query, "playback_id"
        )
        if not playback_id:
            self._json_response(
                400,
                {
                    "error": "missing_param",
                    "message": "Missing required query parameter: playback-id",
                },
            )
            return

        generate_count_raw = self._first_query(query, "generate_count")
        if not generate_count_raw:
            self._json_response(
                400,
                {
                    "error": "missing_param",
                    "message": "Missing required query parameter: generate_count",
                },
            )
            return
        try:
            generate_count = int(generate_count_raw)
        except ValueError:
            self._json_response(
                400,
                {
                    "error": "invalid_param",
                    "message": "generate_count must be an integer greater than 0",
                },
            )
            return
        if generate_count <= 0:
            self._json_response(
                400,
                {
                    "error": "invalid_param",
                    "message": "generate_count must be greater than 0",
                },
            )
            return

        config_raw = self._first_query(query, "config")
        if not config_raw:
            self._json_response(
                400,
                {
                    "error": "missing_param",
                    "message": "Missing required query parameter: config",
                },
            )
            return

        config_path = Path(config_raw).expanduser()
        if not config_path.is_absolute():
            config_path = (self.daemon_settings.repo_root / config_path).resolve()
        if not config_path.exists() or not config_path.is_file():
            self._json_response(
                400,
                {
                    "error": "invalid_param",
                    "message": f"Config file not found: {config_path}",
                },
            )
            return

        workdir = self._first_query(query, "workdir") or self.daemon_settings.default_workdir
        output_dir = self._first_query(query, "output_dir") or self.daemon_settings.default_output_dir
        dynamic_poll = self._first_query(query, "dynamic_poll_seconds")
        dynamic_poll_seconds = self.daemon_settings.dynamic_poll_seconds
        if dynamic_poll:
            try:
                dynamic_poll_seconds = max(1, int(dynamic_poll))
            except ValueError:
                self._json_response(
                    400,
                    {
                        "error": "invalid_param",
                        "message": "dynamic_poll_seconds must be an integer >= 1",
                    },
                )
                return

        command = [
            self.daemon_settings.python_bin,
            "main.py",
            "-c",
            str(config_path),
            "-w",
            workdir,
            "-o",
            output_dir,
            "--dynamic-generation",
            str(generate_count),
            "--playback-device",
            playback_id,
            "--dynamic-poll-seconds",
            str(dynamic_poll_seconds),
        ]

        with self.daemon_state.lock:
            self._refresh_state_locked()
            if self.daemon_state.process is not None:
                self._json_response(
                    409,
                    {
                        "error": "already_running",
                        "message": "A dynamic generation process is already running.",
                        "pid": self.daemon_state.process.pid,
                        "started_at_utc": self.daemon_state.started_at_utc,
                        "last_params": self.daemon_state.last_params,
                    },
                )
                return
            process = subprocess.Popen(command, cwd=str(self.daemon_settings.repo_root))
            self.daemon_state.process = process
            self.daemon_state.started_at_utc = datetime.now(timezone.utc).isoformat()
            self.daemon_state.last_command = list(command)
            self.daemon_state.last_params = {
                "config": str(config_path),
                "playback_id": playback_id,
                "generate_count": generate_count,
                "workdir": workdir,
                "output_dir": output_dir,
                "dynamic_poll_seconds": dynamic_poll_seconds,
            }
            self.daemon_state.last_exit_code = None

        self._json_response(
            200,
            {
                "status": "started",
                "pid": process.pid,
                "started_at_utc": self.daemon_state.started_at_utc,
                "command": command,
                "params": self.daemon_state.last_params,
            },
        )

    def _handle_stop_dynamic(self, query: dict[str, list[str]]) -> None:
        if not self._check_auth(query):
            return

        with self.daemon_state.lock:
            self._refresh_state_locked()
            process = self.daemon_state.process
            if process is None:
                self._json_response(
                    200,
                    {
                        "status": "not_running",
                        "message": "No dynamic generation process is currently running.",
                    },
                )
                return
            pid = process.pid
            process.terminate()

        forced = False
        exit_code: int | None = None
        try:
            exit_code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            exit_code = process.wait(timeout=5)

        with self.daemon_state.lock:
            self.daemon_state.last_exit_code = exit_code
            self.daemon_state.process = None

        self._json_response(
            200,
            {
                "status": "stopped",
                "pid": pid,
                "exit_code": exit_code,
                "forced_kill": forced,
            },
        )

    def _handle_status(self, query: dict[str, list[str]]) -> None:
        if not self._check_auth(query):
            return

        with self.daemon_state.lock:
            self._refresh_state_locked()
            process = self.daemon_state.process
            running = process is not None
            payload = {
                "status": "running" if running else "idle",
                "running": running,
                "pid": process.pid if process else None,
                "started_at_utc": self.daemon_state.started_at_utc,
                "last_exit_code": self.daemon_state.last_exit_code,
                "last_params": self.daemon_state.last_params,
            }
        self._json_response(200, payload)

    def _refresh_state_locked(self) -> None:
        process = self.daemon_state.process
        if process is None:
            return
        code = process.poll()
        if code is None:
            return
        self.daemon_state.last_exit_code = code
        self.daemon_state.process = None

    def _check_auth(self, query: dict[str, list[str]]) -> bool:
        expected = self.daemon_settings.auth_token
        if not expected:
            return True

        provided = self._first_query(query, "token")
        if not provided:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip()

        if provided and hmac.compare_digest(provided, expected):
            return True

        self._json_response(
            401,
            {
                "error": "unauthorized",
                "message": "Provide a valid token via Authorization: Bearer <token> or ?token=<token>.",
            },
        )
        return False

    @staticmethod
    def _first_query(query: dict[str, list[str]], key: str) -> str:
        values = query.get(key) or []
        return str(values[0]).strip() if values else ""

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic generation control daemon for AI Radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on.")
    parser.add_argument(
        "--auth-token",
        default=os.getenv("DYNAMIC_DAEMON_TOKEN", "").strip(),
        help="Token used for endpoint auth (or set DYNAMIC_DAEMON_TOKEN).",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to spawn main.py.",
    )
    parser.add_argument(
        "--workdir",
        default=".tmp",
        help="Default workdir passed to main.py (can be overridden per request).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Default output dir passed to main.py (can be overridden per request).",
    )
    parser.add_argument(
        "--dynamic-poll-seconds",
        type=int,
        default=5,
        help="Default dynamic poll interval passed to main.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    settings = DaemonSettings(
        repo_root=repo_root,
        python_bin=str(args.python_bin),
        default_workdir=str(args.workdir),
        default_output_dir=str(args.output_dir),
        dynamic_poll_seconds=max(1, int(args.dynamic_poll_seconds)),
        auth_token=str(args.auth_token).strip(),
    )
    state = DaemonState()

    DynamicDaemonHandler.daemon_settings = settings
    DynamicDaemonHandler.daemon_state = state

    server = ThreadingHTTPServer((args.host, args.port), DynamicDaemonHandler)
    token_mode = "enabled" if settings.auth_token else "disabled"
    print(
        f"[dynamic-daemon] listening on http://{args.host}:{args.port} auth={token_mode} "
        f"python={settings.python_bin}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dynamic-daemon] shutting down")
    finally:
        with state.lock:
            process = state.process
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            state.process = None
        server.server_close()


if __name__ == "__main__":
    main()
