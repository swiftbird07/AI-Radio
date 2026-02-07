from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class MusicAssistantError(RuntimeError):
    pass


@dataclass
class CommandAttempt:
    command: str
    args: dict[str, Any]
    error: str


class MusicAssistantClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        verify_ssl: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout_seconds = timeout_seconds

    def _ssl_context(self) -> ssl.SSLContext | None:
        if self.verify_ssl:
            return None
        return ssl._create_unverified_context()

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> Any:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self._ssl_context(),
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise MusicAssistantError(
                f"HTTP {exc.code} for {url}: {details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise MusicAssistantError(f"Network error for {url}: {exc.reason}") from exc

        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw

    def call_command(self, command: str, args: dict[str, Any] | None = None) -> Any:
        payload: dict[str, Any] = {"command": command, "args": args or {}}
        data = self._request_json("/api", payload)
        if isinstance(data, dict) and data.get("error"):
            raise MusicAssistantError(
                f"Command '{command}' failed: {data.get('error')}"
            )
        return data

    def try_commands(
        self,
        commands: list[str],
        args_variants: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], Any]:
        attempts: list[CommandAttempt] = []
        for command in [c for c in commands if c]:
            for args in args_variants:
                try:
                    result = self.call_command(command, args)
                    return command, args, result
                except MusicAssistantError as exc:
                    attempts.append(
                        CommandAttempt(command=command, args=args, error=str(exc))
                    )
        errors = "\n".join(
            f"- {a.command} args={a.args}: {a.error}" for a in attempts[-12:]
        )
        raise MusicAssistantError(f"No command variant succeeded.\n{errors}")

