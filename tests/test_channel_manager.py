from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


class _FlakyChannel(BaseChannel):
    name = "flaky"

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(SimpleNamespace(allow_from=["*"]), bus)
        self.start_calls = 0
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.start_calls += 1
        self._running = True
        if self.start_calls == 1:
            self._running = False
            raise RuntimeError("boom")

        self.started.set()
        await self.stopped.wait()

    async def stop(self) -> None:
        self._running = False
        self.stopped.set()

    async def send(self, msg: OutboundMessage) -> None:
        return None


@pytest.mark.asyncio
async def test_start_all_restarts_channel_after_failure() -> None:
    manager = ChannelManager(Config(), MessageBus())
    flaky = _FlakyChannel(manager.bus)
    manager.channels = {"flaky": flaky}

    run_task = asyncio.create_task(manager.start_all())
    await asyncio.wait_for(flaky.started.wait(), timeout=3)

    assert flaky.start_calls >= 2
    assert flaky.is_running is True

    await manager.stop_all()
    await asyncio.wait_for(run_task, timeout=3)

    assert flaky.is_running is False
