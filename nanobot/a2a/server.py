"""Expose a nanobot AgentLoop through the official A2A protocol SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor
from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import A2AConfig


class NanobotAgentExecutor(AgentExecutor):
    """Bridge A2A tasks to an existing nanobot AgentLoop."""

    def __init__(self, agent_loop: Any) -> None:
        self.agent_loop = agent_loop

    async def execute(self, context: Any, event_queue: Any) -> None:
        from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part
        from a2a.server.tasks import TaskUpdater
        from a2a.types import TaskState

        message = context.message
        if message is None:
            raise ValueError("A2A request does not contain a message")

        task = context.current_task or new_task_from_user_message(message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("nanobot is processing the request"),
        )

        query = context.get_user_input().strip()
        if not query:
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                new_text_message("A non-empty text message is required"),
            )
            return

        try:
            response = await self.agent_loop.process_direct(
                query,
                session_key=f"a2a:{task.context_id}",
                channel="a2a",
                chat_id=task.context_id,
            )
        except Exception:
            logger.exception("A2A task {} failed in nanobot", task.id)
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                new_text_message("nanobot failed to process the request"),
            )
            return

        await updater.add_artifact(
            name="nanobot-response",
            parts=[new_text_part(text=response or "", media_type="text/plain")],
        )
        await updater.update_status(
            TaskState.TASK_STATE_COMPLETED,
            new_text_message("Request completed"),
        )

    async def cancel(self, context: Any, event_queue: Any) -> None:
        raise NotImplementedError("A2A task cancellation is not supported")


def build_agent_card(config: A2AConfig):
    """Build the public Agent Card from nanobot configuration."""
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

    skills = [
        AgentSkill(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            input_modes=["text/plain"],
            output_modes=["text/plain"],
            tags=skill.tags,
            examples=skill.examples,
        )
        for skill in config.skills
    ]
    return AgentCard(
        name=config.name,
        description=config.description,
        version=config.version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=config.public_url.rstrip("/"),
                protocol_version="1.0",
            )
        ],
        skills=skills,
    )


def create_a2a_app(agent_loop: Any, config: A2AConfig):
    """Create the Starlette application serving Agent Card and JSON-RPC routes."""
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    card = build_agent_card(config)
    request_handler = DefaultRequestHandler(
        agent_executor=NanobotAgentExecutor(agent_loop),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    async def health(_request):
        return JSONResponse({"status": "ok", "service": "nanobot-a2a"})

    routes = [Route("/healthz", health, methods=["GET"])]
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(request_handler, "/"))
    return Starlette(routes=routes)


def create_a2a_server(agent_loop: Any, config: A2AConfig):
    """Create a configured Uvicorn server for the nanobot A2A app."""
    import uvicorn

    app = create_a2a_app(agent_loop, config)
    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True,
    )
    return uvicorn.Server(uvicorn_config)
