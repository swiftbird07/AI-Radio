from __future__ import annotations

import asyncio
import ssl
import threading
from typing import Any, Callable, Coroutine, TypeVar

from music_assistant_client import MusicAssistantClient as SDKClient
from music_assistant_models.enums import ImageType, MediaType
from music_assistant_models.errors import MusicAssistantError as SDKError
from music_assistant_models.errors import ProviderUnavailableError as SDKProviderUnavailableError
from music_assistant_models.media_items import MediaItemImage

T = TypeVar("T")


class MusicAssistantError(RuntimeError):
    pass


class MusicAssistantProviderUnavailableError(MusicAssistantError):
    pass


class MusicAssistantClient:
    """SDK-only wrapper around music-assistant-client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl

    def _ssl_context(self) -> ssl.SSLContext | None:
        if self.verify_ssl:
            return None
        return ssl._create_unverified_context()

    async def _run_with_client(
        self,
        operation: Callable[[SDKClient], Coroutine[Any, Any, T]],
    ) -> T:
        op_name = getattr(operation, "__name__", "operation")
        try:
            async with SDKClient(
                self.base_url,
                None,
                token=self.api_key,
                ssl_context=self._ssl_context(),
            ) as client:
                return await operation(client)
        except SDKProviderUnavailableError as exc:
            raise MusicAssistantProviderUnavailableError(
                f"{op_name} failed: {exc}"
            ) from exc
        except SDKError as exc:
            raise MusicAssistantError(f"{op_name} failed: {exc}") from exc
        except Exception as exc:
            raise MusicAssistantError(f"{op_name} failed: {exc}") from exc

    def _run(self, operation: Callable[[SDKClient], Coroutine[Any, Any, T]]) -> T:
        async def runner() -> T:
            return await self._run_with_client(operation)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())

        result: dict[str, Any] = {}
        error: dict[str, Exception] = {}

        def _thread_runner() -> None:
            try:
                result["value"] = asyncio.run(runner())
            except Exception as exc:  # pragma: no cover
                error["value"] = exc

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()
        if "value" in error:
            raise error["value"]
        return result["value"]

    @staticmethod
    def _model_to_dict(item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            return item.to_dict()
        if isinstance(item, dict):
            return item
        return {"value": item}

    def get_players(self) -> list[dict[str, Any]]:
        async def op(client: SDKClient) -> list[dict[str, Any]]:
            result = await client.send_command("players/all")
            return [self._model_to_dict(x) for x in result] if isinstance(result, list) else []

        return self._run(op)

    def get_playlist(
        self,
        item_id: str,
        provider_instance_id_or_domain: str,
    ) -> dict[str, Any]:
        async def op(client: SDKClient) -> dict[str, Any]:
            playlist = await client.music.get_playlist(item_id, provider_instance_id_or_domain)
            return self._model_to_dict(playlist)

        return self._run(op)

    def get_playlist_tracks(
        self,
        item_id: str,
        provider_instance_id_or_domain: str,
        page: int = 0,
    ) -> list[dict[str, Any]]:
        async def op(client: SDKClient) -> list[dict[str, Any]]:
            tracks = await client.music.get_playlist_tracks(item_id, provider_instance_id_or_domain, page=page)
            return [self._model_to_dict(x) for x in tracks]

        return self._run(op)

    def create_playlist(
        self,
        name: str,
        provider_instance_or_domain: str | None = None,
    ) -> dict[str, Any]:
        async def op(client: SDKClient) -> dict[str, Any]:
            playlist = await client.music.create_playlist(name, provider_instance_or_domain)
            return self._model_to_dict(playlist)

        return self._run(op)

    def set_playlist_cover(
        self,
        item_id: str | int,
        provider_instance_id_or_domain: str,
        cover_path: str,
        cover_provider: str | None = None,
    ) -> dict[str, Any]:
        async def op(client: SDKClient) -> dict[str, Any]:
            playlist = await client.music.get_playlist(
                str(item_id),
                provider_instance_id_or_domain,
            )
            metadata = playlist.metadata
            metadata.images = [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=cover_path,
                    provider=cover_provider or provider_instance_id_or_domain,
                    remotely_accessible=cover_path.startswith("http://")
                    or cover_path.startswith("https://"),
                )
            ]
            updated = await client.music.update_playlist(
                item_id=item_id,
                update=playlist,
                overwrite=True,
            )
            return self._model_to_dict(updated)

        return self._run(op)

    def add_playlist_tracks(self, db_playlist_id: str | int, uris: list[str]) -> None:
        async def op(client: SDKClient) -> None:
            await client.music.add_playlist_tracks(db_playlist_id, uris)

        self._run(op)

    def get_library_tracks(
        self,
        search: str | None = None,
        limit: int | None = None,
        provider: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        async def op(client: SDKClient) -> list[dict[str, Any]]:
            tracks = await client.music.get_library_tracks(
                search=search,
                limit=limit,
                provider=provider,
            )
            return [self._model_to_dict(x) for x in tracks]

        return self._run(op)

    def get_library_playlists(
        self,
        search: str | None = None,
        limit: int | None = None,
        provider: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        async def op(client: SDKClient) -> list[dict[str, Any]]:
            playlists = await client.music.get_library_playlists(
                search=search,
                limit=limit,
                provider=provider,
            )
            return [self._model_to_dict(x) for x in playlists]

        return self._run(op)

    def remove_playlist(self, item_id: str | int, recursive: bool | None = None) -> None:
        async def op(client: SDKClient) -> None:
            await client.music.remove_playlist(item_id=item_id, recursive=recursive)

        self._run(op)

    def start_sync(self, providers: list[str] | None = None) -> None:
        async def op(client: SDKClient) -> None:
            await client.music.start_sync(media_types=[MediaType.TRACK], providers=providers)

        self._run(op)
