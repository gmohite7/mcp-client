from langgraph_core import StateGraph, Node,START,END
from typing import TypedDict,Dict,Any
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────
class InsightState(TypedDict, total=False):
    user_query: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    tickets: Dict[str, Any]
    insight_draft: Dict[str, Any]
    final_insight: Dict[str, Any]

load_dotenv()


# ─────────────────────────────────────────────
# MCP server definitions (adjust to your setup)
# ─────────────────────────────────────────────
SERVERS = {
    "dynatrace": {
        "command": "uvx",
        "args": ["fastmcp", "run", "dynatrace_mcp:app"],  # TODO: your entrypoint
    },
    "elk": {
        "command": "uvx",
        "args": ["fastmcp", "run", "elk_mcp:app"],        # TODO: your entrypoint
    },
}

# Globals to hold MCP client + tools
_mcp_client: MultiServerMCPClient | None = None
_dyna_tool = None
_elk_tool = None

async def _setup_mcp():
    """Initialise MultiServerMCPClient and pick the tools we need."""
    global _mcp_client, _dyna_tool, _elk_tool
    if _mcp_client is not None:
        return

    _mcp_client = MultiServerMCPClient.from_dict(SERVERS)
    await _mcp_client.__aenter__()
    tools = await _mcp_client.get_tools()

    # TODO: replace these `name` filters with your real MCP tool names
    for t in tools:
        if t.name == "dynatrace_query_timeseries":     # <--- change
            _dyna_tool = t
        if t.name == "elk_search_tickets":             # <--- change
            _elk_tool = t

    if _dyna_tool is None:
        raise RuntimeError("Dynatrace MCP tool not found – check tool name.")
    if _elk_tool is None:
        raise RuntimeError("ELK MCP tool not found – check tool name.")


def call_dynatrace_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper around Dynatrace MCP tool."""
    async def _inner():
        await _setup_mcp()
        return await _dyna_tool.ainvoke(args)
    return asyncio.run(_inner())

def call_elk_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper around ELK MCP tool."""
    async def _inner():
        await _setup_mcp()
        return await _elk_tool.ainvoke(args)
    return asyncio.run(_inner())

# ─────────────────────────────────────────────
# LLM for reasoning / extraction / formatting
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1-mini",    # or your preferred model
    temperature=0.1,
)

# ─────────────────────────────────────────────
# Node 1: Params extractor
# ─────────────────────────────────────────────
PARAMS_SYSTEM_PROMPT = """
You are a parameter extraction agent for a Cloud Insight system.

Given a user query, you MUST output a compact JSON object with:
{
  "resource_type": "vm",
  "metric": "memory_usage_percent",
  "threshold_percent": 80,
  "continuous_hours": 5,
  "lookback_days": 30
}

If the user does not specify some values, infer reasonable defaults.
Do NOT add commentary, return ONLY a JSON object.
"""

def extract_params(state: InsightState) -> InsightState:
    user_query = state["user_query"]

    resp = llm.invoke(
        [
            {"role": "system", "content": PARAMS_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
    )

    # The model should return raw JSON as text
    try:
        params = json.loads(resp.content)
    except Exception:
        raise ValueError(f"LLM did not return valid JSON for params: {resp.content}")

    return {"params": params}

# ─────────────────────────────────────────────
# Node 2: Fetch Dynatrace metrics
# ─────────────────────────────────────────────
def fetch_dynatrace_metrics(state: InsightState) -> InsightState:
    """
    Call Dynatrace MCP with the extracted params.

    You need to adapt the 'args' to match your Dynatrace MCP schema.
    Example idea:
      metric: builtin:host.mem.usage
      entities: HOST or VM IDs (could be wildcarded)
    """
    params = state["params"]

    # TODO: adjust these args to your Dynatrace MCP tool interface
    dynatrace_args = {
        "metric_key": params.get("metric", "builtin:host.mem.usage"),
        "resource_type": params.get("resource_type", "vm"),
        "lookback_days": params.get("lookback_days", 30),
        "rollup_minutes": 5,
    }

    result = call_dynatrace_tool(dynatrace_args)
    return {"metrics": result}

# ─────────────────────────────────────────────
# Node 3: Fetch ELK tickets
# ─────────────────────────────────────────────
def fetch_elk_tickets(state: InsightState) -> InsightState:
    """
    Use ELK MCP to fetch tickets/incidents for enrichment
    in the same time window.
    """
    params = state["params"]

    # TODO: adjust index / query / time filters to your ELK MCP schema
    elk_args = {
        "index": "incident_tickets*",
        "query": "memory OR high memory OR OOM",
        "lookback_days": params.get("lookback_days", 30),
        "size": 100,
    }

    tickets = call_elk_tool(elk_args)
    return {"tickets": tickets}

