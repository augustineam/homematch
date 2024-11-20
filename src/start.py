from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    AIMessageChunk,
)

from pydantic import BaseModel, Field
from typing import Dict, List, Union

from .agent import HomeMatchAgent, cli_achat
from .utils import gradio_add_listing_images
from dotenv import load_dotenv

import gradio as gr
import regex as re
import json
import click
import asyncio


load_dotenv(".env")


@click.group()
@click.option(
    "--vecstore",
    default="./vecstore",
    help="Path to vector store",
    show_default=True,
)
@click.pass_context
def start(ctx: click.Context, vecstore: str):
    """Start HomeMatch AI agent"""
    ctx.obj["vecstore"] = vecstore


@click.command()
@click.pass_context
def terminal(ctx: click.Context):
    """Chat with the HomeMatch agent from the terminal."""

    vecstore = ctx.obj["vecstore"]
    properties_csv = ctx.obj["properties_csv"]

    asyncio.run(cli_achat(vecstore, properties_csv))


@click.command()
@click.option(
    "--port",
    default=8080,
    help="Port to run the server on.",
    show_default=True,
)
@click.option(
    "--host",
    default="localhost",
    help="Host to run the server on.",
    show_default=True,
)
@click.pass_context
def langserve(ctx: click.Context, port: int, host: str):
    """Serve HomeMatch agent as an API endopoint with LangServe and use the `/playground`."""

    from fastapi import FastAPI
    from langserve import add_routes
    from langchain_core.runnables import RunnableLambda

    from src.agent import HomeMatchAgent

    import uvicorn

    app = FastAPI(
        title="HomeMatch Server",
        version="1.0",
        description="A Real State AI Agent to help users find the house of their dreams.",
    )

    # from starlette.routing import Mount
    # from starlette.applications import Starlette
    # from starlette.staticfiles import StaticFiles
    # app.mount("/", app=StaticFiles(directory="ui", html=True), name="HomeMatchUI")

    vecstore = ctx.obj["vecstore"]
    properties_csv = ctx.obj["properties_csv"]

    class InputChat(BaseModel):
        """Input for the chat endpoint."""

        messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
            ...,
            description="The chat messages representing the current conversation.",
        )

    def oup(state: Union[Dict, List]) -> str:
        """Parse agent output and return its response.

        The LangGraph can return either the `agent` (dict) update or
        the `summarize` and `agent` updates (list)
        """

        if isinstance(state, list):
            filterout = [s for s in state if s.get("agent", None)]
            if 0 < len(filterout):
                agent = filterout[0]["agent"]
        elif isinstance(state, dict) and "agent" in state:
            agent = state["agent"]
        else:
            return ""

        messages = [m for m in agent["messages"] if isinstance(m, AIMessage)]
        return messages[-1].content

    config = dict(configurable=dict(thread_id=4))
    agent = HomeMatchAgent(vecstore, properties_csv).pipe(RunnableLambda(oup))
    add_routes(
        app,
        agent.with_config(config).with_types(input_type=InputChat, output_type=str),
        enable_feedback_endpoint=True,
        enable_public_trace_link_endpoint=True,
        playground_type="chat",
    )
    uvicorn.run(app, host=host, port=port)


@click.command()
@click.option(
    "--images-path",
    default="data/images",
    help="Path where the listing images are stored.",
    show_default=True,
)
@click.pass_context
def gradio(ctx: click.Context, images_path: str):
    """Chat with the HomeMatch agent from the gradio UI."""

    # NOTE: To debug the graph's chain
    # from langchain.globals import set_verbose
    # from langchain.callbacks.tracers import ConsoleCallbackHandler

    # set_verbose(True)
    # config = dict(configurable=dict(thread_id=4), callbacks=[ConsoleCallbackHandler()])

    row_pattern = re.compile(r"<<(\d+)>>")

    vecstore = ctx.obj["vecstore"]
    properties_csv = ctx.obj["properties_csv"]
    config = dict(configurable=dict(thread_id=4))

    agent = HomeMatchAgent(
        vecstore,
        properties_csv,
        with_images=True,
        config=config,
    )

    async def invoke_agent(message, history):
        # Unused argument
        history

        input_message = HumanMessage(content=message["text"])

        gathered = ""
        json_chunk = ""
        async for msg, _ in agent.astream(
            dict(messages=[input_message]), stream_mode="messages"
        ):
            if isinstance(msg, AIMessageChunk):

                # FIXME: It is possible we are getting the a chunk with summary json...
                # But what if the Agent sends a similar content that is not the summary?
                # In whispers, he falls
                # Empty nest, broken promise
                # Honor's echo flees.
                if msg.content.startswith("{") or json_chunk:
                    json_chunk = json_chunk + msg.content
                    if msg.content.endswith("}"):
                        try:
                            summary: Dict = json.loads(json_chunk)
                            if "summary" not in summary:
                                gathered = gathered + json_chunk
                            else:
                                print("Summary:\n", "\n".join(summary["summary"]))
                                # FIXME: For some reason the summary is not updated
                                # after adding the json_chunk fix...
                                agent.update_state(config, values=summary)
                                yield gathered
                        except json.decoder.JSONDecodeError:
                            continue
                    else:
                        continue
                else:
                    finish_reason = msg.response_metadata.get("finish_reason", "")
                    gathered = gathered + ("\n" if finish_reason else msg.content)

                # Replaces <<row_number>> pattern with the images in HTML.
                rows = [row for row in row_pattern.findall(gathered)]
                for row in rows:
                    gathered = gathered.replace(
                        f"<<{row}>>", gradio_add_listing_images(images_path, int(row))
                    )

                yield gathered

    demo = gr.ChatInterface(
        invoke_agent, multimodal=True, type="messages", title="HomeMatch Agent"
    )
    demo.launch()


start.add_command(gradio)
start.add_command(langserve)
start.add_command(terminal)