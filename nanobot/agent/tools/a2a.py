"""Outbound A2A peer tools."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from google.protobuf.json_format import MessageToDict
from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.schema import A2APeerConfig


_TERMINAL_ERROR_STATES = {
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}


def _tool_name(peer_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", peer_name).strip("_-").lower()
    if not safe:
        raise ValueError("A2A peer name must contain a letter or number")
    return f"a2a_{safe}_send"[:64]


def _validated_url(url: str) -> tuple[str, str, int, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A2A peer URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("A2A peer URL must not contain credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("A2A peer URL contains an invalid port") from exc
    path = parsed.path.rstrip("/")
    return parsed.scheme, parsed.hostname.lower(), port, path


def _validate_card_interfaces(card: Any, peer_url: str, allow_cross_origin: bool) -> None:
    peer_origin = _validated_url(peer_url)[:3]
    for interface in card.supported_interfaces:
        interface_origin = _validated_url(interface.url)[:3]
        if not allow_cross_origin and interface_origin != peer_origin:
            raise ValueError(
                "Agent Card advertises a cross-origin endpoint; "
                "set allowCrossOriginCard only when that endpoint is trusted"
            )


def _text_parts(parts: list[dict[str, Any]]) -> list[str]:
    return [part["text"] for part in parts if isinstance(part.get("text"), str)]


def _normalize_response(peer_name: str, agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if task := payload.get("task"):
        artifacts = []
        text = []
        for artifact in task.get("artifacts", []):
            artifact_text = _text_parts(artifact.get("parts", []))
            text.extend(artifact_text)
            artifacts.append(
                {
                    "id": artifact.get("artifactId"),
                    "name": artifact.get("name"),
                    "text": "\n".join(artifact_text),
                }
            )
        status = task.get("status", {})
        return {
            "peer": peer_name,
            "agent": agent_name,
            "state": status.get("state", "TASK_STATE_UNSPECIFIED"),
            "taskId": task.get("id"),
            "contextId": task.get("contextId"),
            "text": "\n".join(text),
            "statusMessage": "\n".join(
                _text_parts(status.get("message", {}).get("parts", []))
            ),
            "artifacts": artifacts,
        }

    if message := payload.get("message"):
        return {
            "peer": peer_name,
            "agent": agent_name,
            "state": "MESSAGE",
            "taskId": message.get("taskId"),
            "contextId": message.get("contextId"),
            "text": "\n".join(_text_parts(message.get("parts", []))),
            "artifacts": [],
        }

    return {
        "peer": peer_name,
        "agent": agent_name,
        "state": "UNKNOWN",
        "text": "",
        "artifacts": [],
    }


class A2APeerTool(Tool):
    """Send a text request to one preconfigured A2A agent."""

    def __init__(self, peer_name: str, config: A2APeerConfig) -> None:
        self.peer_name = peer_name
        self.config = config
        self._name = _tool_name(peer_name)
        _validated_url(config.url)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        suffix = f" {self.config.description.strip()}" if self.config.description.strip() else ""
        return (
            f"Send a text request to the configured A2A agent '{self.peer_name}'.{suffix} "
            "Preserve the user's requested time span, classifications, detail level, "
            "output format, totals, and no-duplication constraints. Do not invent dates, "
            "expand a concise request, or request detail the user did not ask for."
        )

    @property
    def relay_response(self) -> bool:
        """Whether a completed peer answer should be returned without another LLM pass."""
        return self.config.relay_response

    @property
    def forward_user_message(self) -> bool:
        """Whether the current user text should replace an LLM-rewritten request."""
        return self.config.forward_user_message

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "The request to send to the remote A2A agent. Faithfully preserve "
                        "the user's scope, duration, requested brevity, and output constraints."
                    ),
                    "minLength": 1,
                    "maxLength": 32000,
                },
                "context_id": {
                    "type": "string",
                    "description": "Optional remote A2A context ID for a follow-up turn",
                    "maxLength": 256,
                },
            },
            "required": ["message"],
        }

    async def execute(self, message: str, context_id: str | None = None, **_kwargs: Any) -> str:
        from a2a.client import A2ACardResolver, ClientConfig, create_client
        from a2a.helpers import new_text_message
        from a2a.types import Role, SendMessageConfiguration, SendMessageRequest

        timeout_s = max(1, min(self.config.timeout_s, 3600))
        max_chars = max(1000, min(self.config.max_response_chars, 100000))
        http_client = httpx.AsyncClient(
            headers=self.config.headers,
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=False,
        )
        client = None
        try:
            async with asyncio.timeout(timeout_s):
                resolver = A2ACardResolver(
                    httpx_client=http_client,
                    base_url=self.config.url,
                    agent_card_path=self.config.agent_card_path,
                )
                card = await resolver.get_agent_card()
                _validate_card_interfaces(
                    card,
                    self.config.url,
                    self.config.allow_cross_origin_card,
                )
                client = await create_client(
                    agent=card,
                    client_config=ClientConfig(
                        streaming=False,
                        httpx_client=http_client,
                        accepted_output_modes=["text/plain"],
                    ),
                )
                request = SendMessageRequest(
                    message=new_text_message(
                        message,
                        context_id=context_id or None,
                        role=Role.ROLE_USER,
                    ),
                    configuration=SendMessageConfiguration(
                        accepted_output_modes=["text/plain"],
                        return_immediately=False,
                    ),
                )
                last_payload = None
                async for chunk in client.send_message(request):
                    last_payload = MessageToDict(chunk)

                if last_payload is None:
                    return f"Error: A2A peer '{self.peer_name}' returned no response"

                result = _normalize_response(
                    self.peer_name,
                    card.name,
                    last_payload,
                )
                encoded = json.dumps(result, ensure_ascii=False)
                if len(encoded) > max_chars:
                    result = {
                        "peer": str(result.get("peer", ""))[:200],
                        "agent": str(result.get("agent", ""))[:200],
                        "state": result.get("state", "UNKNOWN"),
                        "taskId": result.get("taskId"),
                        "contextId": result.get("contextId"),
                        "text": str(result.get("text", ""))[: max_chars // 2],
                        "artifacts": [],
                        "truncated": True,
                    }
                    encoded = json.dumps(result, ensure_ascii=False)
                if result["state"] in _TERMINAL_ERROR_STATES:
                    return f"Error: A2A peer '{self.peer_name}' returned {result['state']}: {encoded}"
                return encoded
        except TimeoutError:
            return f"Error: A2A peer '{self.peer_name}' timed out after {timeout_s}s"
        except Exception as exc:
            return f"Error: A2A peer '{self.peer_name}' call failed: {type(exc).__name__}: {exc}"
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
            if not http_client.is_closed:
                try:
                    await http_client.aclose()
                except Exception:
                    pass


def register_a2a_peer_tools(
    peers: dict[str, A2APeerConfig],
    registry: ToolRegistry,
    local_url: str | None = None,
) -> list[str]:
    """Register one outbound tool per configured peer and return registered names."""
    registered = []
    local_identity = _validated_url(local_url) if local_url else None
    for peer_name, config in peers.items():
        if not config.enabled:
            continue
        try:
            peer_identity = _validated_url(config.url)
            if local_identity is not None and peer_identity == local_identity:
                raise ValueError("peer URL points to this nanobot A2A server")
            tool = A2APeerTool(peer_name, config)
            if registry.has(tool.name):
                raise ValueError(f"tool name collision: {tool.name}")
            registry.register(tool)
            registered.append(tool.name)
        except ValueError as exc:
            logger.error("A2A peer '{}': not registered: {}", peer_name, exc)
    return registered
