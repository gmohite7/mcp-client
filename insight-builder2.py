from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, TypedDict

import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from dotenv import load_dotenv

load_dotenv()

# ---------- LangGraph state ----------

class InsightState(TypedDict, total=False):
    user_query: str
    params: Dict[str, Any]
    dynatrace_plan: Dict[str, Any]
    dynatrace_raw: Dict[str, Any]
    dynatrace_df: pd.DataFrame
    elk_plan: Dict[str, Any]
    elk_raw: Dict[str, Any]
    insight_draft: Dict[str, Any]
    enriched_summary: str
    final_insight: Dict[str, Any]

# ---------- MCP tool plan ----------

class MCPToolPlan(BaseModel):
    """Plan for calling a single MCP tool."""
    server: Literal["dynatrace", "elk"] = Field(
        description="Name of the MCP server (dynatrace or elk)."
    )
    tool_name: str = Field(
        description="Name of the tool to invoke, e.g. 'execute_dql'."
    )
    args: Dict[str, Any] = Field(
        description="Arguments for the tool, e.g. {'dqlStatement': '...'}."
    )
    description: str = Field(
        description="Short explanation of what this call is doing."
    )


# ---------- LLM + MCP client ----------

# llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
llm = ChatOpenAI()

mcp_client = MultiServerMCPClient(
    {
        "dynatrace": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@dynatrace-oss/dynatrace-mcp-server@latest"],
        },
        "elk": {
            "transport": "stdio",
            "command": "python",
            "args": ["./elk_mcp_server.py"],  # <-- your ELK MCP
        },
    }
)

def extract_params(state: InsightState) -> InsightState:
    user_query = state["user_query"]
    prompt = f"""
    You are a parameter extractor for a Cloud Actionable Insight.

    User query:
    {user_query}

    Extract a JSON with:
    {{
      "cloud_provider": "azure",
      "resource_type": "vm",
      "metric_name": "memory_usage",
      "threshold": 0.85,
      "continuous_hours": 5,
      "lookback_days": 30
    }}

    If the user does not specify some values, infer reasonable defaults.
    Return ONLY the JSON object.
    """

    resp = llm.invoke(prompt)
    try:
        params = json.loads(resp.content)
    except Exception:
        raise ValueError(f"Param extractor did not return JSON: {resp.content}")
    return {"params": params}

planner_llm = llm.with_structured_output(MCPToolPlan)

def plan_dynatrace_query(state: InsightState) -> InsightState:
    params = state["params"]
    prompt = f"""
    You are a Dynatrace query planner.

    The user wants to find VMs that have high memory usage for
    {params.get('continuous_hours', 5)} continuous hours
    within the last {params.get('lookback_days', 30)} days.

    The Dynatrace MCP exposes a tool called 'execute_dql' which takes:
      - dqlStatement: string, a valid Dynatrace DQL query.

    You MUST:
    - Set server = "dynatrace".
    - Set tool_name = "execute_dql".
    - Build args.dqlStatement as a valid DQL query that:
        * fetches VM memory usage for the lookback window,
        * buckets into a reasonable time grain (e.g. 5 minutes),
        * returns enough information to compute continuous high-usage periods.

    Do NOT call any tools; just return the MCPToolPlan.
    """
    plan: MCPToolPlan = planner_llm.invoke(prompt)
    return {"dynatrace_plan": plan.dict()}

async def _execute_plan(plan: MCPToolPlan) -> Dict[str, Any]:
    async with mcp_client.session(plan.server) as session:
        tools = await load_mcp_tools(session)
        # Simple match; adjust if tool names differ slightly
        tool = next(t for t in tools if t.name.endswith(plan.tool_name))
        result = await tool.ainvoke(plan.args)
    return result

def execute_dynatrace_plan(state: InsightState) -> InsightState:
    plan = MCPToolPlan(**state["dynatrace_plan"])
    result = asyncio.run(_execute_plan(plan))

    # For first run, print to see the schema:
    # print(json.dumps(result, indent=2))

    # For now, assume rows are under result["data"] or result directly
    rows = result.get("data") or result
    df = pd.DataFrame(rows)

    return {
        "dynatrace_raw": result,
        "dynatrace_df": df,
    }

def plan_elk_query(state: InsightState) -> InsightState:
    params = state["params"]
    bad_vm_ids = (
        state.get("dynatrace_df", pd.DataFrame())
            .get("dt.entity.cloud_vm", pd.Series(dtype=str))
            .dropna()
            .unique()
            .tolist()
    )

    prompt = f"""
    You are an ELK (Elasticsearch) query planner.

    We have identified some VMs with potential high memory problems:
    {bad_vm_ids}

    We want to fetch related incident tickets from an index called
    'incident_tickets*' over the last {params.get('lookback_days', 30)} days.

    The ELK MCP exposes an Elasticsearch query tool that accepts:
    - body: an Elasticsearch JSON query.

    You MUST:
    - Set server = "elk".
    - Set tool_name to the ELK tool that executes a JSON query
      (for now, set it to "execute_query" – you can adjust later).
    - Set args.body as a valid Elasticsearch query that:
        * filters tickets where vm_id is in the above list (if any),
        * limits to last N days.

    If there are no bad_vm_ids, still generate a simple time-filter query.

    Return the MCPToolPlan ONLY.
    """
    plan: MCPToolPlan = planner_llm.invoke(prompt)
    return {"elk_plan": plan.dict()}


def execute_elk_plan(state: InsightState) -> InsightState:
    if "elk_plan" not in state:
        return {}

    plan = MCPToolPlan(**state["elk_plan"])
    result = asyncio.run(_execute_plan(plan))
    return {"elk_raw": result}

def compute_insight(state: InsightState) -> InsightState:
    df = state["dynatrace_df"].copy()
    params = state["params"]
    threshold = float(params.get("threshold", 0.85))
    continuous_hours = float(params.get("continuous_hours", 5))

    # Example: treat avg_value as memory usage fraction (0–1)
    if "avg_value" not in df.columns:
        return {"insight_draft": {"affected_resources": []}}

    df["memory_pct"] = df["avg_value"] * 100.0
    df["above"] = df["memory_pct"] > (threshold * 100.0)

    # Assume data is grouped by vm + time bucket
    # Approximate continuous hours by counting buckets
    # (you can refine later with proper time-diff logic)
    bucket_minutes = 5  # must match your DQL bin()
    group = df.groupby("dt.entity.cloud_vm")["above"].sum().reset_index()
    group["approx_high_hours"] = group["above"] * bucket_minutes / 60.0

    bad = group[group["approx_high_hours"] >= continuous_hours]

    affected_resources = []
    for _, row in bad.iterrows():
        vm_id = row["dt.entity.cloud_vm"]
        vm_data = df[df["dt.entity.cloud_vm"] == vm_id]
        affected_resources.append(
            {
                "resource_id": vm_id,
                "display_name": vm_id,
                "avg_memory_percent": float(vm_data["memory_pct"].mean()),
                "max_memory_percent": float(vm_data["memory_pct"].max()),
                "continuous_violation_hours": float(row["approx_high_hours"]),
                # you can later compute proper windows here
                "occurrence_windows": [],
            }
        )

    return {"insight_draft": {"affected_resources": affected_resources}}

def enrich_insight(state: InsightState) -> InsightState:
    draft = state["insight_draft"]
    elk_raw = state.get("elk_raw")
    params = state["params"]

    table = pd.DataFrame(draft.get("affected_resources", [])).to_markdown(
        index=False
    )

    prompt = f"""
    You are a cloud SRE assistant.

    These VMs have sustained high memory usage:
    {table}

    Related tickets/alerts from ELK:
    {json.dumps(elk_raw, default=str) if elk_raw else "None"}

    In 5–8 sentences:
    - Summarize key patterns,
    - Call out possible business impact,
    - Suggest 3–5 concrete remediation steps.

    Output plain text (no JSON).
    """

    resp = llm.invoke(prompt)
    return {"enriched_summary": resp.content}

def format_insight(state: InsightState) -> InsightState:
    draft = state["insight_draft"]
    summary = state["enriched_summary"]
    params = state["params"]

    prompt = f"""
    Convert the following into a single Cloud Actionable Insight JSON.

    Parameters:
    {json.dumps(params)}

    Affected resources:
    {json.dumps(draft, default=str)}

    Narrative summary:
    {summary}

    Use this JSON structure:

    {{
      "insight_type": "Actionable",
      "category": "Cloud / VM Memory",
      "title": "VMs with sustained high memory usage",
      "summary": "<2–4 sentences summary in simple language>",
      "scope": {{
        "cloud_provider": "<from params>",
        "resource_type": "vm"
      }},
      "evidence": {{
        "time_window_days": <int>,
        "threshold_percent": <float>,
        "continuous_hours": <float>,
        "affected_vms": [
          {{
            "vm_id": "<id>",
            "avg_memory_percent": <float>,
            "max_memory_percent": <float>,
            "continuous_violation_hours": <float>
          }}
        ]
      }},
      "recommended_actions": [
        "<action 1>",
        "<action 2>"
      ],
      "severity": "High|Medium|Low"
    }}

    Return ONLY valid JSON, nothing else.
    """

    resp = llm.invoke(prompt)
    final_insight = json.loads(resp.content)
    return {"final_insight": final_insight}

def build_graph():
    graph = StateGraph(InsightState)

    # LLM nodes
    graph.add_node("extract_params", extract_params)
    graph.add_node("plan_dynatrace_query", plan_dynatrace_query)
    graph.add_node("plan_elk_query", plan_elk_query)
    graph.add_node("enrich_insight", enrich_insight)
    graph.add_node("format_insight", format_insight)

    # Pure Python nodes
    graph.add_node("execute_dynatrace_plan", execute_dynatrace_plan)
    graph.add_node("execute_elk_plan", execute_elk_plan)
    graph.add_node("compute_insight", compute_insight)

    graph.set_entry_point("extract_params")

    graph.add_edge("extract_params", "plan_dynatrace_query")
    graph.add_edge("plan_dynatrace_query", "execute_dynatrace_plan")
    graph.add_edge("execute_dynatrace_plan", "compute_insight")
    graph.add_edge("compute_insight", "plan_elk_query")
    graph.add_edge("plan_elk_query", "execute_elk_plan")
    graph.add_edge("execute_elk_plan", "enrich_insight")
    graph.add_edge("enrich_insight", "format_insight")
    graph.add_edge("format_insight", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    initial: InsightState = {
        "user_query": "Find Azure VMs that had high memory usage "
                      "for at least 5 continuous hours in the last 30 days "
                      "and generate a cloud actionable insight."
    }
    final_state = app.invoke(initial)
    print(json.dumps(final_state["final_insight"], indent=2))

