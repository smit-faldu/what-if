"""
LangGraph nodes for the What If content creation pipeline.

Graph flow:
  START → brainstorm_ideas → filter_ideas → select_idea
        → generate_content → save_output → END

Note: script writing and image prompt generation are merged into
`generate_content` because each dialog line owns its image prompt.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, OUTPUT_DIR,
    TTS_ENABLED, TTS_MODEL, TTS_SPEAKER, TTS_LANGUAGE, TTS_INSTRUCT,
)
from pipeline.state import (
    WhatIfState,
    IdeaList,
    SelectedIdea,
    VideoScript,
)
from pipeline.prompts import (
    BRAINSTORM_SYSTEM, BRAINSTORM_USER,
    SELECT_SYSTEM, SELECT_USER,
    SCRIPT_SYSTEM, SCRIPT_USER,
    FLUX_STYLE_PREFIX,
)
from pipeline.vector_memory import IdeaVectorMemory


# ── Shared LLM factory ────────────────────────────────────────────────────────

def _make_llm(temperature: float = 0.9) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


# ── Node 1: Brainstorm Ideas ──────────────────────────────────────────────────

def brainstorm_ideas(state: WhatIfState) -> dict:
    """
    Ask Gemini for `idea_count` fresh What If candidates.
    Queries recent ideas from vector memory and injects them to avoid repeats.
    """
    count = state.get("idea_count", 10)
    llm = _make_llm().with_structured_output(IdeaList)

    memory = IdeaVectorMemory()
    past_ideas = memory.get_recent_ideas(limit=25)

    if past_ideas:
        avoid_ideas_clause = (
            "CRITICAL: Do NOT generate ideas that are identical or highly similar to these ideas which have already been used:\n"
            + "\n".join(f"- {idea}" for idea in past_ideas)
        )
    else:
        avoid_ideas_clause = ""

    messages = [
        SystemMessage(content=BRAINSTORM_SYSTEM),
        HumanMessage(
            content=BRAINSTORM_USER.format(
                count=count,
                avoid_ideas_clause=avoid_ideas_clause,
            )
        ),
    ]

    result: IdeaList = llm.invoke(messages)
    return {"candidates": result.ideas}


# ── Node 2: Filter Ideas (semantic deduplication) ─────────────────────────────

def filter_ideas(state: WhatIfState) -> dict:
    """
    Embed each candidate with the local all-MiniLM-L6-v2 model and reject
    those too similar (cosine similarity > threshold) to past ideas in Supabase.
    If ALL candidates are rejected, keep the least-similar one as a fallback.
    """
    candidates: list[str] = state["candidates"]
    memory = IdeaVectorMemory()

    passed: list[str] = []
    scored: list[tuple[float, str]] = []  # (similarity, idea)

    for idea in candidates:
        sim = memory.get_similarity(idea)
        score = sim if sim is not None else 0.0
        scored.append((score, idea))

        if not memory.is_too_similar(idea):
            passed.append(idea)

    if passed:
        return {"filtered_candidates": passed}

    # Fallback: keep the least similar idea so the pipeline never deadlocks
    scored.sort(key=lambda x: x[0])
    fallback = scored[0][1]
    print(
        f"\n⚠️  All {len(candidates)} candidates were too similar to past ideas. "
        f"Using least-similar as fallback: '{fallback}'\n"
    )
    return {"filtered_candidates": [fallback]}


# ── Node 3: Select Best Idea ──────────────────────────────────────────────────

def select_idea(state: WhatIfState) -> dict:
    """Pick the single best idea from the filtered candidates."""
    filtered: list[str] = state["filtered_candidates"]
    llm = _make_llm(temperature=0.5).with_structured_output(SelectedIdea)

    ideas_text = "\n".join(f"- {idea}" for idea in filtered)
    messages = [
        SystemMessage(content=SELECT_SYSTEM),
        HumanMessage(content=SELECT_USER.format(ideas=ideas_text)),
    ]

    result: SelectedIdea = llm.invoke(messages)
    return {
        "selected_idea": result.idea,
        "selection_reason": result.reason,
    }


# ── Node 4: Generate Content (script + image prompts together) ────────────────

def generate_content(state: WhatIfState) -> dict:
    """
    Generate the full video script as 7-10 dialog lines.
    Each dialog line contains:
    - text        : TTS-ready voiceover (10-20 words)
    - flux_prompt : Pixar/Disney Flux image prompt for that exact moment
    - dialog_type : hook | body | closer

    Script + image prompts are generated in a single structured LLM call
    because each line owns its visual.
    """
    idea: str = state["selected_idea"]
    llm = _make_llm(temperature=0.85).with_structured_output(VideoScript)

    messages = [
        SystemMessage(content=SCRIPT_SYSTEM),
        HumanMessage(content=SCRIPT_USER.format(idea=idea)),
    ]

    result: VideoScript = llm.invoke(messages)

    # Guarantee every flux_prompt starts with the hardcoded style prefix
    for dialog in result.dialogs:
        if not dialog.flux_prompt.strip().startswith("Pixar"):
            dialog.flux_prompt = FLUX_STYLE_PREFIX + dialog.flux_prompt

    return {"script": result.model_dump()}


# ── Node 5: Generate TTS Audio ────────────────────────────────────────────────

def generate_tts(state: WhatIfState) -> dict:
    """
    Convert every dialog line's text into speech using Qwen3-TTS.

    Behaviour:
    - Skips gracefully (returns empty dict) when TTS_ENABLED=false in config/env.
    - Tries CUDA first; automatically falls back to CPU if no GPU is available.
    - Generates one .wav per dialog line saved alongside the final output.
    - Concatenates all clips into a single merged audio.wav.
    - Returns the path to the merged audio file via state["tts_audio_path"].

    NOTE: The output folder hasn't been created yet at this stage — audio files
    are buffered in-memory (numpy arrays) and written by save_output.
    The raw wavs and sample rate are stored temporarily in state["script"]["_tts"].
    """
    if not TTS_ENABLED:
        print("\n⏭️  TTS_ENABLED=false — skipping audio generation.")
        return {}

    # Lazy import so machines without qwen-tts installed can still run
    # the rest of the pipeline (they'll hit a clear ImportError here only).
    try:
        import torch
        import numpy as np
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        print(
            f"\n⚠️  TTS packages not installed ({exc}). "
            "Run: pip install qwen-tts soundfile\n"
            "Skipping audio generation."
        )
        return {}

    dialogs: list[dict] = state["script"]["dialogs"]

    # ── Device selection: CUDA → CPU fallback ────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype  = torch.float16
        print(f"\n🎙️  TTS: using GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        dtype  = torch.float32
        print("\n🎙️  TTS: no GPU found — falling back to CPU (may be slow)")

    print(f"   Loading model: {TTS_MODEL} ...")
    tts_model = Qwen3TTSModel.from_pretrained(
        TTS_MODEL,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    # ── Generate audio per dialog line ───────────────────────────────────────
    clips: list  = []   # list of 1-D numpy float32 arrays
    sample_rate: int | None = None

    for d in dialogs:
        line_num = d["line_number"]
        text     = d["text"]
        print(f"   Generating line {line_num}/{len(dialogs)}: {text[:60]}...")

        # Sanitize text to prevent Qwen3-TTS from hallucinating non-speech sounds (like laughing)
        # on non-standard Unicode symbols like em-dashes and horizontal ellipses.
        clean_text = text.replace("—", ", ").replace("…", "... ").replace(" - ", ", ")

        wavs, sr = tts_model.generate_custom_voice(
            text=clean_text,
            language=TTS_LANGUAGE,
            speaker=TTS_SPEAKER,
            instruct=TTS_INSTRUCT,
        )
        
        # Maximize volume / normalize clip to prevent it from being too quiet
        clip_wav = wavs[0]
        max_val = np.max(np.abs(clip_wav))
        if max_val > 0:
            clip_wav = (clip_wav / max_val) * 0.95
            
        clips.append(clip_wav)
        if sample_rate is None:
            sample_rate = sr

    # ── Stitch clips together with a short silence gap (0.3 s) ──────────────
    import numpy as np
    silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
    merged  = np.concatenate(
        [arr for clip in clips for arr in (clip, silence)]
    )

    # Normalize final merged track to peak at 0.98 for maximum clarity and volume
    max_val_merged = np.max(np.abs(merged))
    if max_val_merged > 0:
        merged = (merged / max_val_merged) * 0.98

    # Store raw data in script dict so save_output can write the files
    script = dict(state["script"])
    script["_tts"] = {
        "clips":       [c.tolist() for c in clips],  # JSON-serialisable
        "merged":      merged.tolist(),
        "sample_rate": sample_rate,
    }
    print(f"   ✅ TTS done — {len(clips)} clips, merged duration: "
          f"{len(merged)/sample_rate:.1f}s")
    return {"script": script}


# ── Node 6: Save Output ───────────────────────────────────────────────────────

def save_output(state: WhatIfState) -> dict:
    """
    1. Store the chosen idea in Supabase vector memory (for future deduplication).
    2. Write a timestamped output folder with:
       - dialogs.md     : numbered dialog lines + image prompts (human-readable)
       - tts_script.txt : expressive text (one line per dialog, symbols intact)
       - line_XX.wav    : individual TTS clips (if TTS was run)
       - audio.wav      : merged TTS audio (if TTS was run)
       - data.json      : full structured output
    """
    idea: str = state["selected_idea"]
    script: dict = state["script"]
    run_id: str = state["run_id"]
    dialogs: list[dict] = script.get("dialogs", [])

    # ── 1. Persist idea in Supabase ──────────────────────────────────────────
    memory = IdeaVectorMemory()
    memory.add_idea(idea)

    # ── 2. Create timestamped output folder ──────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = idea[:40].replace(" ", "_").replace("?", "").replace("/", "-")
    folder: Path = OUTPUT_DIR / f"{ts}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)

    # ── 3. Write dialogs.md ──────────────────────────────────────────────────
    TYPE_EMOJI = {"hook": "🎣", "body": "🎬", "closer": "🤯"}
    lines_md = [
        f"# {idea}\n",
        f"> Run ID: `{run_id}`  |  Est. duration: "
        f"**{script.get('total_duration_estimate', '?')} seconds**\n",
        f"> Total dialogs: **{len(dialogs)}**\n",
        "---\n",
    ]
    for d in dialogs:
        emoji = TYPE_EMOJI.get(d["dialog_type"], "📌")
        lines_md.append(
            f"## {emoji} Line {d['line_number']} [{d['dialog_type'].upper()}]\n"
        )
        lines_md.append(f"**TTS:** {d['text']}\n")
        lines_md.append(f"\n**Flux Prompt:**\n```\n{d['flux_prompt']}\n```\n")
        lines_md.append("---\n")
    (folder / "dialogs.md").write_text("\n".join(lines_md), encoding="utf-8")

    # ── 4. Write tts_script.txt (expressive text — one line per dialog) ───────
    # Symbols like — and … are preserved. Safe to paste directly into Colab.
    tts_lines = [d["text"] for d in dialogs]
    (folder / "tts_script.txt").write_text(
        "\n".join(tts_lines), encoding="utf-8"
    )

    # ── 5. Write TTS audio files (if generate_tts ran successfully) ──────────
    tts_data    = script.get("_tts")
    audio_path  = None
    if tts_data:
        try:
            import numpy as np
            import soundfile as sf

            sr     = tts_data["sample_rate"]
            clips  = [np.array(c, dtype=np.float32) for c in tts_data["clips"]]
            merged = np.array(tts_data["merged"],   dtype=np.float32)

            # Individual clips
            for i, clip in enumerate(clips, start=1):
                sf.write(str(folder / f"line_{i:02d}.wav"), clip, sr)

            # Merged audio
            audio_path = str(folder / "audio.wav")
            sf.write(audio_path, merged, sr)
            print(f"\n🔊 Audio saved → {audio_path}")

            # Remove raw TTS blob before saving data.json (too large)
            script = {k: v for k, v in script.items() if k != "_tts"}
        except Exception as exc:
            print(f"\n⚠️  Could not write audio files: {exc}")

    # ── 6. Write data.json (full structured output) ───────────────────────────
    data = {
        "run_id": run_id,
        "idea": idea,
        "selection_reason": state.get("selection_reason", ""),
        "script": script,
        "audio_path": audio_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (folder / "data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "output_path":    str(folder),
        "tts_audio_path": audio_path,
    }
