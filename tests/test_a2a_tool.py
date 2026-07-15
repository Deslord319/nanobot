import asyncio
import json
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from nanobot.a2a.server import create_a2a_app
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.a2a import (
    A2APeerTool,
    _validate_card_interfaces,
    register_a2a_peer_tools,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import A2AConfig, A2APeerConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest


class _FakeAgentLoop:
    def __init__(self, response: str = "REMOTE_A2A_OK", delay_s: float = 0) -> None:
        self.response = response
        self.delay_s = delay_s
        self.calls = []

    async def process_direct(self, content, **kwargs):
        self.calls.append((content, kwargs))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self.response


class _RequireHeaderMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, expected: str) -> None:
        super().__init__(app)
        self.expected = expected

    async def dispatch(self, request, call_next):
        if request.headers.get("authorization") != self.expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


@asynccontextmanager
async def _running_a2a_server(agent, *, require_header: str | None = None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    app = create_a2a_app(agent, A2AConfig(public_url=url))
    if require_header:
        app.add_middleware(_RequireHeaderMiddleware, expected=require_header)
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="critical", lifespan="off", access_log=False)
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    try:
        yield url
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        sock.close()


@pytest.mark.asyncio
async def test_peer_tool_calls_remote_agent_with_headers_and_context() -> None:
    agent = _FakeAgentLoop()
    auth = "Bearer test-token"
    async with _running_a2a_server(agent, require_header=auth) as url:
        tool = A2APeerTool(
            "research-agent",
            A2APeerConfig(url=url, headers={"Authorization": auth}, timeout_s=5),
        )
        first = json.loads(await tool.execute(message="first request"))
        second = json.loads(
            await tool.execute(message="follow up", context_id=first["contextId"])
        )

    assert tool.name == "a2a_research-agent_send"
    assert first["state"] == "TASK_STATE_COMPLETED"
    assert first["text"] == "REMOTE_A2A_OK"
    assert second["contextId"] == first["contextId"]
    assert agent.calls[1][1]["session_key"] == f"a2a:{first['contextId']}"


@pytest.mark.asyncio
async def test_peer_tool_timeout_isolated_as_tool_error() -> None:
    async with _running_a2a_server(_FakeAgentLoop(delay_s=2)) as url:
        tool = A2APeerTool("slow", A2APeerConfig(url=url, timeout_s=1))
        result = await tool.execute(message="slow request")

    assert result == "Error: A2A peer 'slow' timed out after 1s"


def test_registration_blocks_self_peer_and_tool_name_collisions() -> None:
    registry = ToolRegistry()
    peers = {
        "self": A2APeerConfig(url="http://127.0.0.1:18791"),
        "worker one": A2APeerConfig(url="http://127.0.0.1:19001"),
        "worker_one": A2APeerConfig(url="http://127.0.0.1:19002"),
    }

    names = register_a2a_peer_tools(
        peers,
        registry,
        local_url="http://127.0.0.1:18791/",
    )

    assert names == ["a2a_worker_one_send"]
    assert registry.has("a2a_worker_one_send")


def test_cross_origin_agent_card_is_rejected_by_default() -> None:
    card = AgentCard(
        name="redirecting-agent",
        description="test",
        version="1",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(
                url="http://127.0.0.1:19999",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
    )

    with pytest.raises(ValueError, match="cross-origin"):
        _validate_card_interfaces(card, "http://127.0.0.1:18888", False)

    _validate_card_interfaces(card, "http://127.0.0.1:18888", True)


def test_peer_tool_requires_faithful_user_constraint_forwarding() -> None:
    tool = A2APeerTool(
        "finance",
        A2APeerConfig(url="http://127.0.0.1:18793", description="Finance expert"),
    )

    assert "Preserve the user's requested time span" in tool.description
    message_description = tool.parameters["properties"]["message"]["description"]
    assert "requested brevity" in message_description


@pytest.mark.asyncio
async def test_configured_peer_response_is_relayed_without_second_llm_pass(
    tmp_path: Path,
) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    tool_call = ToolCallRequest(
        id="call1",
        name="a2a_finance_send",
        arguments={"message": "扩写后的详细逐日请求"},
    )
    provider.chat = AsyncMock(
        return_value=LLMResponse(content="", tool_calls=[tool_call])
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        a2a_peers={
            "finance": A2APeerConfig(
                url="http://127.0.0.1:18793",
                forward_user_message=True,
                relay_response=True,
            )
        },
    )
    peer_tool = loop.tools.get("a2a_finance_send")
    assert peer_tool is not None
    peer_tool.execute = AsyncMock(
        return_value=json.dumps(
            {
                "state": "TASK_STATE_COMPLETED",
                "text": "专家原始答案：净总计 ¥100。",
            },
            ensure_ascii=False,
        )
    )

    original_request = "未来20天，简洁展示，不要交叉重复"
    final_content, tools_used, messages = await loop._run_agent_loop(
        [{"role": "user", "content": original_request}]
    )

    assert final_content == "专家原始答案：净总计 ¥100。"
    assert tools_used == ["a2a_finance_send"]
    assert provider.chat.await_count == 1
    peer_tool.execute.assert_awaited_once_with(message=original_request)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == final_content
