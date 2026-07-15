from a2a.helpers import new_text_message
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueueLegacy
from a2a.types import Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict
from starlette.testclient import TestClient

from nanobot.a2a.server import NanobotAgentExecutor, create_a2a_app
from nanobot.config.schema import A2AConfig


class _FakeAgentLoop:
    def __init__(self) -> None:
        self.calls = []

    async def process_direct(self, content, **kwargs):
        self.calls.append((content, kwargs))
        return "NANOBOT_A2A_OK"


def test_agent_card_and_health_routes() -> None:
    config = A2AConfig(name="nanobot-test", public_url="http://127.0.0.1:18791")
    app = create_a2a_app(_FakeAgentLoop(), config)

    with TestClient(app) as client:
        health = client.get("/healthz")
        card = client.get("/.well-known/agent-card.json")

    assert health.json() == {"status": "ok", "service": "nanobot-a2a"}
    assert card.status_code == 200
    assert card.json()["name"] == "nanobot-test"
    assert card.json()["supportedInterfaces"][0]["protocolVersion"] == "1.0"


async def test_executor_routes_text_to_nanobot_session() -> None:
    agent = _FakeAgentLoop()
    executor = NanobotAgentExecutor(agent)
    request = SendMessageRequest(message=new_text_message("hello a2a", role=Role.ROLE_USER))
    context = RequestContext(ServerCallContext(), request=request)
    queue = EventQueueLegacy()

    await executor.execute(context, queue)
    events = [MessageToDict(await queue.dequeue_event()) for _ in range(4)]

    assert agent.calls[0][0] == "hello a2a"
    assert agent.calls[0][1]["channel"] == "a2a"
    assert agent.calls[0][1]["session_key"].startswith("a2a:")
    assert any(
        event.get("artifact", {}).get("parts", [{}])[0].get("text") == "NANOBOT_A2A_OK"
        for event in events
    )
