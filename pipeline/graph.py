"""
LangGraph StateGraph for the What If content creation pipeline.

Topology (linear — each node feeds into the next):
  START
    │
    ▼
  brainstorm_ideas     ← Gemini generates N raw "What If" candidates
    │
    ▼
  filter_ideas         ← Local all-MiniLM embeddings + Supabase pgvector similarity check
    │
    ▼
  select_idea          ← Gemini picks the single best idea
    │
    ▼
  generate_content     ← Gemini writes 7-10 dialog lines, each with TTS text + Flux image prompt
    │
    ▼
  generate_tts         ← Qwen3-TTS converts each line to speech; merges into audio.wav (CUDA→CPU)
    │
    ▼
  generate_images      ← Calls self-hosted Flux API; saves image_01.jpg … image_NN.jpg
    │
    ▼
  save_output          ← Stores idea in Supabase; writes dialogs.md + tts_script.txt + audio.wav + images + data.json
    │
    ▼
  END
"""

import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config import DB_PATH
from pipeline.state import WhatIfState
from pipeline.nodes import (
    brainstorm_ideas,
    filter_ideas,
    select_idea,
    generate_content,
    generate_tts,
    generate_images,
    save_output,
)

# Keep a module-level connection so the checkpointer stays alive
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def build_graph():
    """
    Construct and compile the What If pipeline graph.

    Returns:
        Compiled LangGraph app with SQLite checkpointer for state persistence.
    """
    graph = StateGraph(WhatIfState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("brainstorm_ideas", brainstorm_ideas)
    graph.add_node("filter_ideas", filter_ideas)
    graph.add_node("select_idea", select_idea)
    graph.add_node("generate_content", generate_content)
    graph.add_node("generate_tts", generate_tts)
    graph.add_node("generate_images", generate_images)
    graph.add_node("save_output", save_output)

    # ── Wire edges (linear flow) ──────────────────────────────────────────────
    graph.add_edge(START, "brainstorm_ideas")
    graph.add_edge("brainstorm_ideas", "filter_ideas")
    graph.add_edge("filter_ideas", "select_idea")
    graph.add_edge("select_idea", "generate_content")
    graph.add_edge("generate_content", "generate_tts")
    graph.add_edge("generate_tts", "generate_images")
    graph.add_edge("generate_images", "save_output")
    graph.add_edge("save_output", END)

    # ── Compile with SQLite checkpointer ──────────────────────────────────────
    app = graph.compile(checkpointer=_checkpointer)
    return app
