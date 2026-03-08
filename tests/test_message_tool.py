import pytest

from nanobot.agent.tools.message import MessageTool


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_suppresses_duplicate_send_in_same_turn() -> None:
    sent = []

    async def _send(msg):
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    tool.set_context("telegram", "123")
    tool.start_turn()

    result1 = await tool.execute(content="hello   world")
    result2 = await tool.execute(content="hello world")

    assert "Message sent" in result1
    assert result2 == "Skipped duplicate message to telegram:123"
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_message_tool_allows_distinct_send_in_same_turn() -> None:
    sent = []

    async def _send(msg):
        sent.append(msg)

    tool = MessageTool(send_callback=_send)
    tool.set_context("telegram", "123")
    tool.start_turn()

    await tool.execute(content="step 1 started")
    await tool.execute(content="step 2 done")

    assert len(sent) == 2
