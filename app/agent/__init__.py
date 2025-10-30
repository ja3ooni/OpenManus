from app.agent.base import BaseAgent


# Lazy imports to avoid dependency issues
def _get_browser_agent():
    from app.agent.browser import BrowserAgent

    return BrowserAgent


def _get_mcp_agent():
    from app.agent.mcp import MCPAgent

    return MCPAgent


def _get_react_agent():
    from app.agent.react import ReActAgent

    return ReActAgent


def _get_swe_agent():
    from app.agent.swe import SWEAgent

    return SWEAgent


def _get_toolcall_agent():
    from app.agent.toolcall import ToolCallAgent

    return ToolCallAgent


def _get_agent_pool():
    from app.agent.pool import (
        AgentPool,
        PoolConfig,
        get_agent_pool,
        shutdown_agent_pool,
    )

    return AgentPool, PoolConfig, get_agent_pool, shutdown_agent_pool


__all__ = [
    "BaseAgent",
]
