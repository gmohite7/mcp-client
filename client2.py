# app.py — Minimal MCP + Streamlit chat (correct tool message ordering, no filler rendered)

import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage




# SYSTEM_PROMPT = (
#     "You have access to tools. When you choose to call a tool, do not narrate status updates. "
#     "After tools run, return only a concise final answer."
# )

SYSTEM_PROMPT = """
You are an operator assistant with access to multiple MCP tools.

IMPORTANT RULES:

1. When the user asks anything about Dynatrace data (metrics or logs),
   you MUST use the Dynatrace MCP tool called 'execute_dql'
   (or whichever Dynatrace tool you see listed that runs DQL).

2. That tool has an argument like:
     { "dqlStatement": "<Dynatrace DQL string>" }

   You must write the DQL yourself based on the user's natural language.
   There is NO separate "generate DQL" tool. You are the one generating the DQL.

3. Example pattern for logs:
   - Fetch 2 ERROR logs in the last day:

     dqlStatement:
     \"\"\"
     fetch logs
     | filter loglevel == "ERROR"
     | filter timestamp >= now() - 1d
     | limit 2
     \"\"\"

4. When you choose to call a tool, do NOT narrate "I am calling a tool".
   Just return a tool call. After tools run, respond with a concise final answer
   explaining what you found.

5. If the user does NOT ask for Dynatrace, use other tools or answer directly.
"""

st.set_page_config(page_title="MCP Chat", page_icon="🧰", layout="centered")
st.title("🧰 MCP Chat")

load_dotenv()

# One-time init
if "initialized" not in st.session_state:
    # 1) LLM
    st.session_state.llm = ChatOpenAI(model="gpt-5")

    # 2) MCP tools
    st.session_state.client = MultiServerMCPClient(SERVERS)
    tools = asyncio.run(st.session_state.client.get_tools())
    st.session_state.tools = tools
    st.session_state.tool_by_name = {t.name: t for t in tools}

    # 3) Bind tools
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    # 4) Conversation state
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.initialized = True

# Render chat history (skip system + tool messages; hide intermediate AI with tool_calls)
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        # Skip assistant messages that contain tool_calls (intermediate “fetching…”)
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # ToolMessage and SystemMessage are not rendered as bubbles

# Chat input
user_text = st.chat_input("Type a message…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    # First pass: let the model decide whether to call tools
    first = asyncio.run(st.session_state.llm_with_tools.ainvoke(st.session_state.history))
    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # No tools → show & store assistant reply
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)
    else:
        # ── IMPORTANT ORDER ──
        # 1) Append assistant message WITH tool_calls (do NOT render)
        st.session_state.history.append(first)

        # 2) Execute requested tools and append ToolMessages (do NOT render)
        tool_msgs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            tool = st.session_state.tool_by_name[name]
            print(f"Invoking tool {name} with args: {args}")
            res = asyncio.run(tool.ainvoke(args))
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(res)))

        st.session_state.history.extend(tool_msgs)

        # 3) Final assistant reply using tool outputs → render & store
        final = asyncio.run(st.session_state.llm.ainvoke(st.session_state.history))
        with st.chat_message("assistant"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))
