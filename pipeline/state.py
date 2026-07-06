"""
Pydantic models (structured output schemas) and the LangGraph state TypedDict
for the What If content creation pipeline.

Script structure: 7-10 fixed dialog lines, each with its own TTS text + image prompt.
This maps 1:1 with TTS clips and image frames for production.
"""

from __future__ import annotations
from typing import Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ── Structured Output Schemas ────────────────────────────────────────────────

class IdeaList(BaseModel):
    """Raw brainstorm output — a list of What If idea candidates."""
    ideas: list[str] = Field(
        description=(
            "List of 'What If' video ideas. Each should be a single, "
            "clear question focused on day-to-day human life."
        )
    )


class SelectedIdea(BaseModel):
    """The single best idea chosen from filtered candidates."""
    idea: str = Field(
        description="The chosen 'What If' idea, phrased as a short question."
    )
    reason: str = Field(
        description=(
            "Why this idea is the most relatable, curiosity-inducing, "
            "and suitable for a 40-60 second video."
        )
    )


class DialogLine(BaseModel):
    """
    A single dialog line — the atomic unit of the video.

    Maps directly to:
    - 1 TTS audio clip (text field)
    - 1 image frame  (flux_prompt field)
    """
    line_number: int = Field(
        description="Sequential number starting at 1.",
        ge=1,
        le=10,
    )
    dialog_type: Literal["hook", "body", "closer"] = Field(
        description=(
            "'hook' for the opening lines, "
            "'body' for the main content, "
            "'closer' for the final punch + CTA."
        )
    )
    text: str = Field(
        description=(
            "The voiceover line for this moment — spoken aloud by TTS. "
            "Must be 10-20 words. Short, punchy, complete sentence. "
            "Use em-dashes (—) and ellipses (…) for emotional pauses. "
            "Natural hesitation words (heh, huh, honestly?) allowed sparingly. "
            "NEVER use bracket-style tags like [chuckles] or [sighs]."
        )
    )
    flux_prompt: str = Field(
        description=(
            "Full Flux image generation prompt for this specific dialog line. "
            "Must start with the Pixar/Disney style prefix, then describe "
            "exactly what is visible in the frame for this line's content."
        )
    )


class VideoScript(BaseModel):
    """
    Complete video script as a sequence of 7-10 fixed dialog lines.

    Structure:
    - 1-2  HOOK lines    : \"What if...\" opener + first shocking consequence
    - 4-6  BODY lines    : funny/absurd/real ripple effects
    - 1-2  CLOSER lines  : biggest mind-blow + CTA

    Each line = 1 TTS clip + 1 image. Total = 7-10 clips.
    """
    dialogs: list[DialogLine] = Field(
        description="Ordered list of 7-10 dialog lines covering the full video arc.",
        min_length=7,
        max_length=10,
    )
    total_duration_estimate: int = Field(
        description=(
            "Estimated total voiceover duration in seconds "
            "(sum of all lines read at normal TTS pace). Target: 40-60 seconds."
        ),
        ge=35,
        le=70,
    )


# ── LangGraph State ──────────────────────────────────────────────────────────

class WhatIfState(TypedDict):
    """Shared state flowing through the What If LangGraph pipeline."""
    run_id: str                          # UUID for this pipeline run
    idea_count: int                      # how many candidates to brainstorm
    candidates: list[str]                # raw ideas from Gemini
    filtered_candidates: list[str]       # ideas that passed similarity filter
    selected_idea: str                   # the winning idea text
    selection_reason: str                # why it was chosen
    script: Optional[dict]               # VideoScript.model_dump() — includes dialogs[]
    output_path: Optional[str]           # path to the saved output folder
    tts_audio_path: Optional[str]        # path to the merged TTS audio (.wav)
    image_paths: Optional[list[str]]     # paths to generated images (one per dialog line)
    subtitle_paths: Optional[dict]       # {"srt": str, "json": str} — Whisper subtitle files

