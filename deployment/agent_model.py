"""MLflow models-from-code definition (Task 2.1).

For Bonus B (agents.deploy()), this file wraps the graph as a ChatAgent
so it has the proper schema that Agent Framework expects.
"""

from __future__ import annotations

import os
from typing import Any, Generator, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever

import tools as _tools_pkg


_REQUIRED_ENV_VARS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL")
_missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
if _missing:
    raise OSError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Set these in the serving endpoint's environment_vars (see "
        "deployment/deploy.py) or in your local .env for local testing."
    )


_MCP_SERVER_PATH = os.path.join(os.path.dirname(_tools_pkg.__file__), "mcp_server.py")


def _build_graph_lazy():
    # Build the runtime graph lazily so importing this module (during MLflow
    # model logging) does not attempt network calls to MCP or Vector Search.
    return build_graph(
        llm=get_chat_llm(),
        retriever=get_retriever(),
        tools=load_mcp_tools(_MCP_SERVER_PATH),
    )


class AnalystChatAgent(ChatAgent):
    """Wrap the AnalystState graph to match Agent Framework ChatAgent interface.

    The graph is built lazily on first `predict`/`predict_stream` call to avoid
    performing network calls at import time (which breaks `mlflow.langchain.log_model`).
    """

    def __init__(self, graph=None):
        self.graph = graph
    
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Invoke the graph and return messages as ChatAgentResponse."""
        # Lazily build the graph if not already built (safe for MLflow log time)
        if self.graph is None:
            self.graph = _build_graph_lazy()

        # Convert ChatAgentMessage list to dict format for the graph
        message_dicts = self._convert_messages_to_dict(messages)
        
        # Invoke graph with full AnalystState initialization
        result = self.graph.invoke({
            "messages": message_dicts,
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "next_agent": "",
            "final_answer": "",
        })
        
        # Extract messages from result state and convert LangChain messages to ChatAgentMessage format
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        import uuid
        
        out_messages = []
        for msg in result["messages"]:
            # Map LangChain message type to role
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, dict):
                # Already a dict, extract role if present
                role = msg.get("role", "user")
            else:
                role = "user"  # Default fallback
            
            # Extract content
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            
            out_messages.append(ChatAgentMessage(
                id=str(uuid.uuid4()),
                role=role,
                content=content
            ))
        
        return ChatAgentResponse(messages=out_messages)
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Stream graph execution as ChatAgentChunk deltas."""
        # Lazily build the graph if not already built
        if self.graph is None:
            self.graph = _build_graph_lazy()

        message_dicts = self._convert_messages_to_dict(messages)
        
        request = {
            "messages": message_dicts,
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "next_agent": "",
            "final_answer": "",
        }
        
        for event in self.graph.stream(request, stream_mode="updates"):
            for node_data in event.values():
                if "messages" in node_data:
                    for msg in node_data["messages"]:
                        yield ChatAgentChunk(**{"delta": msg})


# Enable MLflow tracing for LangChain/LangGraph calls
mlflow.langchain.autolog()

# Wrap and register the agent (graph will be built lazily at runtime)
served_model = AnalystChatAgent()
mlflow.models.set_model(served_model)

