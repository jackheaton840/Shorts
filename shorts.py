#!/usr/bin/env python3
"""
shorts.py - turn one or more YouTube videos into vertical shorts with subtitles.

Pipeline:
    1. Probe the video metadata
    2. Download the video
    3. Transcribe with word-level timestamps (faster-whisper)
    4. Optionally analyze sampled video frames with a local vision model (Ollama)
    5. Generate candidate clips, then rank them with a second AI pass
    6. Cut each clip, make it 9:16 with a blurred background, and burn in animated subtitles

Output structure:
    shorts_output/
        <video_folder>/
            transcript.txt
            vision_notes.json
            ai_reply.txt
            clips/
                01_title.mp4
                01_title.ass
            work/
                downloaded source + cached transcript + frame samples

Only use this on videos you own or have permission to reuse.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_TEXT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_VISION_MODEL = "llava:7b"
SCRIPT_DIR = Path(__file__).resolve().parent
FFMPEG = "ffmpeg"


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def safe_name(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return (text[:limit] or "clip")


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def to_seconds(t: str) -> float:
    secs = 0.0
    for part in t.strip().split(":"):
        secs = secs * 60 + float(part)
    return secs


def parse_time(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return to_seconds(str(v).strip().rstrip("sS"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_loads_loose(text: str, default: Any = None) -> Any:
    if default is None:
        default = []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first valid JSON object/array in the text.
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    for start in sorted(starts):
        if text[start] == "{":
            end = text.rfind("}")
        else:
            end = text.rfind("]")
        if end <= start:
            continue
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
    return default


def unique_folder(base_dir: Path, stem: str) -> Path:
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        alt = base_dir / f"{stem}_{i}"
        if not alt.exists():
            return alt
        i += 1


def write_text_safe(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------
# Step 0: sanity checks
# --------------------------------------------------------------------------
def has_subtitles_filter(exe: str) -> bool:
    """True if this FFmpeg can burn in subtitles (needs libass)."""
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(r"\ssubtitles\s", out) is not None


def find_ffmpeg() -> None:
    """Pick an FFmpeg that can draw captions; fall back to one bundled with pip."""
    global FFMPEG
    system = shutil.which("ffmpeg")
    if system and has_subtitles_filter(system):
        FFMPEG = system
        return

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        try:
            os.chmod(bundled, 0o755)
        except OSError:
            pass
        if has_subtitles_filter(bundled):
            print("  Your system FFmpeg can't draw captions; using the bundled one instead.")
            FFMPEG = bundled
            return
    except ImportError:
        pass

    if system:
        FFMPEG = system
        print("  WARNING: your FFmpeg can't draw captions. Fix: pip install imageio-ffmpeg")
        return

    sys.exit("FFmpeg isn't installed. Run: pip install imageio-ffmpeg")


def check_setup(need_key: bool) -> None:
    find_ffmpeg()
    if need_key and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY isn't set. For a free option use --provider ollama, "
            "or pick the times yourself with --clip 1:20-2:05"
        )


# --------------------------------------------------------------------------
# Metadata + download
# --------------------------------------------------------------------------
def probe_info(url: str) -> Dict[str, Any]:
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def download(url: str, work_dir: Path) -> Tuple[Path, str, Dict[str, Any]]:
    import yt_dlp

    opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if not shutil.which("ffmpeg"):
        opts["ffmpeg_location"] = FFMPEG

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    try:
        path = Path(info["requested_downloads"][0]["filepath"])
    except (KeyError, IndexError):
        path = work_dir / f"{info['id']}.mp4"

    return path, info.get("title", "video"), info


# --------------------------------------------------------------------------
# Step 1: transcribe (cached, so re-runs on the same video are instant)
# --------------------------------------------------------------------------
_WHISPER_MODELS: Dict[str, Any] = {}


def transcribe(video: Path, model_size: str) -> List[Dict[str, Any]]:
    cache = video.with_suffix(".transcript.json")
    if cache.exists():
        print("  Using cached transcript.")
        return json.loads(cache.read_text(encoding="utf-8"))

    from faster_whisper import WhisperModel

    model = _WHISPER_MODELS.get(model_size)
    if model is None:
        model = _WHISPER_MODELS[model_size] = WhisperModel(
            model_size, device="auto", compute_type="auto"
        )
    seg_iter, info = model.transcribe(str(video), word_timestamps=True, vad_filter=True)
    total = info.duration or 1

    segments = []
    for s in seg_iter:
        words = [
            {"w": w.word.strip(), "s": w.start, "e": w.end}
            for w in (s.words or [])
            if w.word.strip()
        ]
        segments.append(
            {"start": s.start, "end": s.end, "text": s.text.strip(), "words": words}
        )
        print(f"  Transcribed {s.end / total:5.1%}", end="\r")
    print()

    cache.write_text(json.dumps(segments), encoding="utf-8")
    return segments


def write_readable_transcript(segments: Sequence[Dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(
        "\n".join(f"[{mmss(s['start'])}] {s['text']}" for s in segments),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Ollama / Claude helpers
# --------------------------------------------------------------------------
def ask_claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def ask_ollama(prompt: str, model: str) -> str:
    import urllib.error
    import urllib.request

    needed = len(prompt) // 3 + 1500
    ctx = 4096 * math.ceil(needed / 4096)
    if ctx > 16384:
        print(
            "  Warning: this transcript is long for a local model; "
            "the results may be poor. Try a shorter video or --clip."
        )
        ctx = 16384

    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": ctx, "temperature": 0.3},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            payload = json.loads(r.read())
            return payload["message"]["content"]
    except urllib.error.HTTPError as e:
        sys.exit(f"Ollama returned an error ({e.code}). Have you run: ollama pull {model} ?")
    except urllib.error.URLError:
        sys.exit("Couldn't reach Ollama. Open the Ollama app (or run: ollama serve) and try again.")


def ask_model(prompt: str, provider: str, ollama_model: str) -> str:
    if provider == "ollama":
        return ask_ollama(prompt, ollama_model)
    return ask_claude(prompt)


def ask_ollama_vision(prompt: str, image_paths: Sequence[Path], model: str) -> str:
    import urllib.error
    import urllib.request

    images_b64 = []
    for p in image_paths:
        images_b64.append(base64.b64encode(p.read_bytes()).decode("ascii"))

    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
        }
    ).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        payload = json.loads(r.read())
        return payload["message"]["content"]


# --------------------------------------------------------------------------
# Step 2: vision analysis
# --------------------------------------------------------------------------
def extract_vision_frames(video: Path, frames_dir: Path, sample_seconds: int) -> List[Tuple[float, Path]]:
    ensure_dir(frames_dir)
    pattern = str(frames_dir / "frame_%04d.jpg")
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps=1/{sample_seconds}",
        "-q:v",
        "5",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:] or "ffmpeg failed to extract frames")

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    return [(i * sample_seconds, p) for i, p in enumerate(frames)]


def chunked(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def analyze_vision(
    video: Path,
    work_dir: Path,
    model: str,
    sample_seconds: int = 8,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Sample frames and ask a local vision model for scene notes."""
    frames_dir = work_dir / "vision_frames"
    cache = work_dir / "vision_notes.json"
    if cache.exists():
        try:
            cached_notes = json.loads(cache.read_text(encoding="utf-8"))
            # An empty cache usually means the model was unavailable on an earlier run.
            # Do not keep treating that failed attempt as a successful cached result.
            if isinstance(cached_notes, list) and cached_notes:
                return cached_notes
        except (json.JSONDecodeError, OSError):
            pass

    try:
        frames = extract_vision_frames(video, frames_dir, sample_seconds)
    except Exception as e:
        print(f"  Vision analysis skipped: {e}")
        return []

    if not frames:
        print("  Vision analysis skipped: no frames were extracted.")
        return []

    notes: List[Dict[str, Any]] = []
    for batch in chunked(frames, batch_size):
        times = ", ".join(f"{t:.0f}s" for t, _ in batch)
        prompt = f"""
You are analyzing sampled frames from a video for short-form clip selection.

Return ONLY JSON in this exact shape:
{{
  "frames": [
    {{
      "time": 12,
      "hook": "brief description of what is visually interesting",
      "energy": 1,
      "scene_change": true,
      "text_on_screen": "any readable text or empty string",
      "why_it_matters": "one short sentence"
    }}
  ]
}}

Rules:
- Use the frame times exactly as provided.
- energy is 1-5, where 5 means visually intense / highly engaging.
- Focus on motion, people, reaction faces, visual novelty, scene changes, demonstrations, graphics, or on-screen text.
- Keep each hook short.
- Do not mention every tiny detail.
- If nothing is interesting, say so plainly.

Frame times in this batch: {times}
""".strip()

        try:
            reply = ask_ollama_vision(prompt, [p for _, p in batch], model)
            parsed = json_loads_loose(reply, default={})
            batch_notes = parsed.get("frames", []) if isinstance(parsed, dict) else []
            for item, (t, _) in zip(batch_notes, batch):
                if not isinstance(item, dict):
                    continue
                notes.append(
                    {
                        "time": float(item.get("time", t)),
                        "hook": str(item.get("hook", "")).strip(),
                        "energy": int(item.get("energy", 0) or 0),
                        "scene_change": bool(item.get("scene_change", False)),
                        "text_on_screen": str(item.get("text_on_screen", "")).strip(),
                        "why_it_matters": str(item.get("why_it_matters", "")).strip(),
                    }
                )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            if e.code == 404:
                print(
                    f"  Vision model '{model}' was not found by Ollama (HTTP 404). "
                    f"Run: ollama pull {model}\n  Ollama response: {detail}"
                )
                print("  Skipping the remaining vision batches; transcript-based selection will continue.")
                break
            print(f"  Vision model skipped a batch (HTTP {e.code}): {detail or e.reason}")
        except Exception as e:
            print(f"  Vision model skipped a batch: {e}")

    # Only persist successful results. A [] cache would prevent a later retry after
    # the user installs the vision model.
    if notes:
        cache.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    else:
        try:
            cache.unlink(missing_ok=True)
        except OSError:
            pass
    return notes


def summarize_vision_notes(notes: Sequence[Dict[str, Any]], limit: int = 20) -> str:
    if not notes:
        return "No vision notes available."
    rows = []
    top = sorted(
        notes,
        key=lambda x: (int(x.get("energy", 0) or 0), bool(x.get("scene_change"))),
        reverse=True,
    )[:limit]
    for note in sorted(top, key=lambda x: x.get("time", 0)):
        bits = [f"{note.get('time', 0):.0f}s"]
        if note.get("energy"):
            bits.append(f"energy {note['energy']}")
        if note.get("scene_change"):
            bits.append("scene change")
        hook = note.get("hook", "")
        if hook:
            bits.append(hook)
        txt = note.get("text_on_screen", "")
        if txt:
            bits.append(f"text: {txt}")
        rows.append(" | ".join(bits))
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Step 3: choose the clips
# --------------------------------------------------------------------------
EXAMPLE_ITEM = '{"start_line": 12, "end_line": 21, "title": "Short hook title", "reason": "why this works"}'


def build_prompt(
    segments: Sequence[Dict[str, Any]],
    vision_notes: Sequence[Dict[str, Any]],
    n: int,
    min_s: int,
    max_s: int,
    as_object: bool,
) -> str:
    transcript = "\n".join(
        f"[{i}] ({s['start']:.0f}s) {s['text']}" for i, s in enumerate(segments)
    )
    total = segments[-1]["end"] if segments else 0
    example = '{"clips": [' + EXAMPLE_ITEM + ']}' if as_object else '[' + EXAMPLE_ITEM + ']'
    vision_text = summarize_vision_notes(vision_notes)

    return f"""You are editing a long video into short vertical clips for social media.

Below is a numbered transcript. Each line shows its line number in [brackets] and the time in seconds when it starts. The video is {total:.0f} seconds long.

Vision notes from sampled frames:
{vision_text}

Pick the {n} most engaging moments. Each moment is a run of consecutive lines, from start_line to end_line (inclusive).

Rules:
- Each moment must last between {min_s} and {max_s} seconds in total. Use the start times shown to work out how many lines that takes.
- Start on the first line of a complete thought and end on a line that finishes a sentence.
- Prefer a strong hook at the start: a surprising claim, a story, a strong opinion, or clear advice. Avoid intros, outros, sponsor reads and small talk.
- Prefer moments that match the vision notes if they look active, visual, emotional, or high-energy.
- Moments must not overlap.
- Use only line numbers that appear in the transcript.

Respond with ONLY JSON, no other text, like:
{example}

Transcript:
{transcript}"""


def parse_clips(text: str) -> List[Dict[str, Any]]:
    data = json_loads_loose(text, default=[])
    if isinstance(data, dict):
        if "clips" in data:
            data = data["clips"]
        elif any(k.startswith("start") for k in data):
            data = [data]
        else:
            data = next((v for v in data.values() if isinstance(v, list)), [])
    return data if isinstance(data, list) else []


def resolve_clips(
    raw: Sequence[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
    min_s: int,
    max_s: int,
) -> List[Dict[str, Any]]:
    """Turn the AI's answer into real clips. Accepts line numbers (preferred) or
    times, fixes lengths that are too short or long, and drops overlaps."""
    if not segments:
        return []
    last = len(segments) - 1
    clips, used = [], []

    for c in raw:
        if not isinstance(c, dict):
            continue
        try:
            if "start_line" in c and "end_line" in c:
                a, b = int(float(c["start_line"])), int(float(c["end_line"]))
            else:
                ts = parse_time(c.get("start", c.get("start_time")))
                te = parse_time(c.get("end", c.get("end_time")))
                a = min(range(len(segments)), key=lambda i: abs(segments[i]["start"] - ts))
                b = min(range(len(segments)), key=lambda i: abs(segments[i]["end"] - te))
        except (TypeError, ValueError):
            continue

        a, b = max(0, min(a, last)), max(0, min(b, last))
        if b < a:
            a, b = b, a

        def dur() -> float:
            return segments[b]["end"] - segments[a]["start"]

        while dur() > max_s and b > a:
            b -= 1
        while dur() < min_s and b < last:
            b += 1
        while dur() > max_s and b > a:
            b -= 1

        if dur() < 10:
            continue

        start, end = segments[a]["start"], segments[b]["end"]
        if any(start < e and end > s for s, e in used):
            continue
        used.append((start, end))
        clips.append(
            {
                "start": start,
                "end": end,
                "title": c.get("title", "clip"),
                "reason": c.get("reason", ""),
            }
        )
    return clips


def fallback_clips(segments: Sequence[Dict[str, Any]], n: int, min_s: int, max_s: int) -> List[Dict[str, Any]]:
    """No AI needed: pick the stretches with the densest speech."""
    target = (min_s + max_s) / 2
    cands = []
    for a in range(len(segments)):
        start = segments[a]["start"]
        b = a
        while b < len(segments) - 1 and segments[b]["end"] - start < target:
            b += 1
        end = segments[b]["end"]
        if end - start < min_s:
            continue
        words = sum(len(s["words"]) for s in segments[a : b + 1])
        cands.append((words / max(0.1, (end - start)), start, end))
    cands.sort(reverse=True)

    clips, used = [], []
    for _, start, end in cands:
        if any(start < e and end > s for s, e in used):
            continue
        used.append((start, end))
        clips.append(
            {
                "start": start,
                "end": end,
                "title": f"clip {len(clips) + 1}",
                "reason": "automatic pick (busiest speech)",
            }
        )
        if len(clips) == n:
            break
    return sorted(clips, key=lambda c: c["start"])


def manual_clips(specs: Sequence[str]) -> List[Dict[str, Any]]:
    clips = []
    for spec in specs:
        try:
            a, b = spec.split("-")
            start, end = to_seconds(a), to_seconds(b)
            if end <= start:
                raise ValueError
        except ValueError:
            sys.exit(f"Couldn't read --clip {spec!r}. Use a format like 1:20-2:05")
        clips.append(
            {"start": start, "end": end, "title": f"clip {len(clips) + 1}", "reason": "chosen by you"}
        )
    return clips


def make_candidate_windows(
    segments: Sequence[Dict[str, Any]],
    vision_notes: Sequence[Dict[str, Any]],
    min_s: int,
    max_s: int,
    target_count: int,
) -> List[Dict[str, Any]]:
    """Create candidate clip windows from transcript density and vision events."""
    if not segments:
        return []

    candidates: List[Dict[str, Any]] = []
    total_end = segments[-1]["end"]
    target = (min_s + max_s) / 2
    half = target / 2

    def snap_window(start_t: float, end_t: float) -> Tuple[int, int]:
        start_t = max(0.0, start_t)
        end_t = min(total_end, end_t)
        a = min(range(len(segments)), key=lambda i: abs(segments[i]["start"] - start_t))
        b = min(range(len(segments)), key=lambda i: abs(segments[i]["end"] - end_t))
        if b < a:
            a, b = b, a
        while b < len(segments) - 1 and (segments[b]["end"] - segments[a]["start"]) < min_s:
            b += 1
        while b > a and (segments[b]["end"] - segments[a]["start"]) > max_s:
            b -= 1
        return a, b

    seen = set()

    def add_candidate(a: int, b: int, source: str) -> None:
        if a < 0 or b >= len(segments) or b < a:
            return
        start, end = segments[a]["start"], segments[b]["end"]
        key = (round(start, 1), round(end, 1))
        if key in seen:
            return
        seen.add(key)
        transcript = " ".join(s["text"] for s in segments[a : b + 1]).strip()
        if not transcript:
            return
        local_vision = [
            n for n in vision_notes if start - 2 <= float(n.get("time", 0)) <= end + 2
        ]
        visual_bits = []
        for n in local_vision[:4]:
            bits = [f"{float(n.get('time', 0)):.0f}s"]
            if n.get("energy"):
                bits.append(f"energy {n['energy']}")
            if n.get("hook"):
                bits.append(str(n["hook"]))
            if n.get("text_on_screen"):
                bits.append(f"text: {n['text_on_screen']}")
            visual_bits.append(" | ".join(bits))
        candidates.append(
            {
                "start": start,
                "end": end,
                "source": source,
                "transcript": transcript[:1200],
                "vision": visual_bits,
            }
        )

    # Dense-speech windows from the transcript.
    for a in range(0, len(segments), max(1, len(segments) // max(1, target_count * 2))):
        start = segments[a]["start"]
        a2, b2 = snap_window(start, start + target)
        add_candidate(a2, b2, "dense speech")

    # Windows centered on visually interesting moments.
    for n in sorted(vision_notes, key=lambda x: int(x.get("energy", 0) or 0), reverse=True):
        if int(n.get("energy", 0) or 0) < 3 and not n.get("scene_change"):
            continue
        t = float(n.get("time", 0))
        a2, b2 = snap_window(t - half, t + half)
        add_candidate(a2, b2, "vision note")

    # If we still don't have enough, just scan forward.
    i = 0
    while len(candidates) < max(10, target_count * 4) and i < len(segments):
        start = segments[i]["start"]
        a2, b2 = snap_window(start, start + target)
        add_candidate(a2, b2, "fallback scan")
        i += max(1, len(segments) // 12)

    candidates.sort(key=lambda c: (c["start"], c["end"]))
    return candidates[: max(12, target_count * 6)]


def rank_candidates_with_model(
    candidates: Sequence[Dict[str, Any]],
    n: int,
    provider: str,
    ollama_model: str,
) -> Tuple[List[Dict[str, Any]], str]:
    if not candidates:
        return []

    blocks = []
    for i, c in enumerate(candidates, 1):
        vision = "\n".join(f"    - {v}" for v in c.get("vision", [])[:4]) or "    - none"
        blocks.append(
            f"""[{i}] start={c['start']:.2f}s end={c['end']:.2f}s
source: {c.get('source', 'unknown')}
transcript: {c.get('transcript', '')}
vision:
{vision}"""
        )

    prompt = f"""You are selecting the best short-form clips from candidate windows.

Choose the {n} best candidates for a vertical short-form video.
Prefer:
- a strong hook in the first seconds
- clear complete thoughts
- moments that feel visually active or emotionally strong
- clips that are not repetitive
- clips that do not overlap

Return ONLY JSON in this exact format:
{{"clips":[{{"id":1,"score":98,"title":"short hook title","reason":"short reason"}}]}}

Candidates:
{chr(10).join(blocks)}
"""
    text = ask_model(prompt, provider, ollama_model)

    parsed = parse_clips(text)

    chosen: List[Dict[str, Any]] = []
    used: List[Tuple[float, float]] = []
    for item in sorted(parsed, key=lambda x: int(x.get("score", 0) or 0), reverse=True):
        try:
            idx = int(item.get("id", 0)) - 1
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        c = candidates[idx]
        start, end = float(c["start"]), float(c["end"])
        if any(start < e and end > s for s, e in used):
            continue
        used.append((start, end))
        chosen.append(
            {
                "start": start,
                "end": end,
                "title": item.get("title", c.get("source", "clip")),
                "reason": item.get("reason", ""),
            }
        )
        if len(chosen) == n:
            break

    if not chosen:
        # If the model returned something malformed, fall back to the first non-overlapping candidates.
        for c in candidates:
            start, end = float(c["start"]), float(c["end"])
            if any(start < e and end > s for s, e in used):
                continue
            used.append((start, end))
            chosen.append(
                {"start": start, "end": end, "title": c.get("source", "clip"), "reason": "fallback candidate"}
            )
            if len(chosen) == n:
                break

    return chosen, text


def pick_clips(segments: Sequence[Dict[str, Any]], args: argparse.Namespace, work_dir: Path) -> List[Dict[str, Any]]:
    if args.clip:
        return manual_clips(args.clip)

    as_object = args.provider == "ollama"
    vision_notes: List[Dict[str, Any]] = []

    if args.use_vision:
        try:
            vision_notes = analyze_vision(
                video=args.video_path,
                work_dir=work_dir,
                model=args.vision_model,
                sample_seconds=args.vision_sample_seconds,
                batch_size=args.vision_batch_size,
            )
            print(f"  Vision notes generated: {len(vision_notes)}")
        except Exception as e:
            print(f"  Vision analysis unavailable: {e}")

    candidates = make_candidate_windows(
        segments=segments,
        vision_notes=vision_notes,
        min_s=args.min,
        max_s=args.max,
        target_count=args.clips,
    )

    if not args.two_pass:
        candidates = []

    if candidates:
        reply, raw_text = rank_candidates_with_model(
            candidates=candidates,
            n=args.clips,
            provider=args.provider,
            ollama_model=args.ollama_model,
        )
        write_text_safe(work_dir / "ai_reply.txt", raw_text)
        write_text_safe(work_dir / "selected_clips.json", json.dumps(reply, indent=2))
        clips = resolve_clips(reply, segments, args.min, args.max)
        if clips:
            return clips

    prompt = build_prompt(segments, vision_notes, args.clips, args.min, args.max, as_object)
    text = ask_model(prompt, args.provider, args.ollama_model)
    write_text_safe(work_dir / "ai_reply.txt", text)

    clips = resolve_clips(parse_clips(text), segments, args.min, args.max)
    if not clips:
        print("  The AI didn't give usable clips. Falling back to the busiest stretches of speech.")
        clips = fallback_clips(segments, args.clips, args.min, args.max)
    return clips


# --------------------------------------------------------------------------
# Step 4a: subtitles (ASS format, one highlighted word at a time)
# --------------------------------------------------------------------------
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t: float) -> str:
    t = max(t, 0)
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_subtitles(segments: Sequence[Dict[str, Any]], start: float, end: float, path: Path, words_per_chunk: int = 3) -> None:
    words = []
    for seg in segments:
        for w in seg["words"]:
            if w["s"] >= start - 0.05 and w["e"] <= end + 0.05:
                words.append(
                    {
                        "w": re.sub(r"[{}\\]", "", w["w"]).upper(),
                        "s": max(w["s"] - start, 0),
                        "e": w["e"] - start,
                    }
                )

    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= words_per_chunk or re.search(r"[.?!,]$", w["w"]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    lines = [ASS_HEADER]
    for chunk in chunks:
        for i, w in enumerate(chunk):
            t_start = w["s"]
            t_end = chunk[i + 1]["s"] if i + 1 < len(chunk) else w["e"]
            parts = []
            for j, other in enumerate(chunk):
                if j == i:
                    parts.append(
                        f"{{\\1c&H00FFFF&\\fscx112\\fscy112}}{other['w']}{{\\1c&HFFFFFF&\\fscx100\\fscy100}}"
                    )
                else:
                    parts.append(other["w"])
            lines.append(
                f"Dialogue: 0,{ass_time(t_start)},{ass_time(t_end)},Default,,0,0,0,,{' '.join(parts)}"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Step 4b: render with FFmpeg
# --------------------------------------------------------------------------
def render(video: Path, start: float, end: float, ass_name: str, out_name: str, out_dir: Path) -> None:
    base = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=40:6[b];"
        "[fg]scale=1080:-2[f];"
        "[b][f]overlay=(W-w)/2:(H-h)/2"
    )

    variants = [
        (f"{base},subtitles=filename={ass_name}[v]", True),
        (f"{base},ass=filename={ass_name}[v]", True),
        (f"{base}[v]", False),
    ]

    first_error = ""
    for filters, has_captions in variants:
        cmd = [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{end - start:.2f}",
            "-i",
            str(video.resolve()),
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            out_name,
        ]
        result = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
        if result.returncode == 0:
            if not has_captions:
                print("  WARNING: FFmpeg couldn't add captions, so this clip has none.")
                print("  FFmpeg said:", first_error.strip()[-600:])
            return
        if not first_error:
            first_error = result.stderr

    sys.exit("FFmpeg failed:\n" + first_error[-1500:])




# --------------------------------------------------------------------------
# Upload helpers
# --------------------------------------------------------------------------
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def normalize_upload_targets(values: Optional[Sequence[str]]) -> List[str]:
    targets: List[str] = []
    if not values:
        return targets
    for value in values:
        for part in str(value).split(","):
            part = part.strip().lower()
            if part:
                targets.append(part)
    if any(t == "all" for t in targets):
        return ["youtube", "tiktok", "instagram"]
    out: List[str] = []
    for t in targets:
        if t not in out:
            out.append(t)
    return out


def clip_upload_metadata(source_title: str, clip: Dict[str, Any], platform: str) -> Dict[str, Any]:
    clip_title = str(clip.get("title") or "clip").strip()
    reason = str(clip.get("reason") or "").strip()
    base = clip_title if clip_title else source_title
    short_source = re.sub(r"\s+", " ", source_title).strip()
    if platform == "youtube":
        title = f"{base} #shorts" if "#shorts" not in base.lower() else base
        title = title[:95].strip()
        description_parts = [short_source]
        description_parts.append("#shorts")
        tags = []
        for word in re.findall(r"[A-Za-z0-9_]+", short_source.lower()):
            if len(word) >= 4 and word not in tags:
                tags.append(word)
        if "shorts" not in tags:
            tags.append("shorts")
        tags = tags[:10]
        return {"title": title, "description": "\n\n".join(description_parts), "tags": tags}

    if platform == "tiktok":
        title = base[:150].strip()
        description_parts = [short_source]
        if reason:
            description_parts.append(reason)
        return {"title": title, "description": "\n\n".join(description_parts)}

    if platform == "instagram":
        caption_parts = [base]
        if reason:
            caption_parts.append(reason)
        caption_parts.append("#shorts")
        return {"caption": "\n\n".join(caption_parts)[:2200]}

    return {}


# --------------------------------------------------------------------------
# YouTube title / description / hashtags (written by the same AI that picks clips)
# --------------------------------------------------------------------------
GENERIC_HASHTAGS = {
    "shorts", "short", "youtube", "youtubeshorts", "video", "viral", "fyp",
    "foryou", "foryoupage", "trending", "subscribe", "explore",
}
KEYWORD_STOPWORDS = {
    "about", "actually", "after", "again", "always", "around", "because", "before", "being",
    "could", "doing", "every", "first", "getting", "going", "little", "maybe", "never",
    "other", "people", "probably", "really", "right", "should", "something", "still", "thing",
    "things", "think", "their", "there", "these", "those", "though", "through", "wanted",
    "where", "which", "while", "would", "yeah", "gonna", "kinda", "anyway",
}


def clip_transcript(segments: List[Dict[str, Any]], start: float, end: float) -> str:
    words = [
        w["w"]
        for seg in segments
        for w in seg.get("words", [])
        if w["s"] >= start - 0.05 and w["e"] <= end + 0.05
    ]
    if words:
        return " ".join(words)
    return " ".join(s["text"] for s in segments if s["start"] >= start - 0.05 and s["end"] <= end + 0.05)


def build_metadata_prompt(source_title: str, clip_title: str, clip_text: str) -> str:
    return f"""You write YouTube Shorts titles, descriptions and hashtags.

Source video title: {source_title}
Working title for this clip: {clip_title}

Transcript of the clip (what is actually said):
---
{clip_text[:2500]}
---

Write metadata for this one clip. Rules:
- Base everything ONLY on the transcript. Do not invent facts, names, numbers or claims that are not in it. Do not promise anything the clip does not deliver.
- title: 40-70 characters, specific, and makes someone curious to watch. Plain sentence case, no ALL CAPS, no hashtags, no quotation marks around it.
- description: 1-2 short sentences (under 250 characters) saying what the clip is about. No hashtags and no links.
- hashtags: 3-5 relevant single-word or CamelCase hashtags about the topic, people or themes. Do not include #Shorts, #fyp, #viral or #youtube.
- tags: 6-10 search keywords or short phrases, without the # symbol.
- Write in the same language as the transcript.

Respond with ONLY JSON, no other text:
{{"title": "...", "description": "...", "hashtags": ["#Example"], "tags": ["example keyword"]}}"""


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return {}
        try:
            data = json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def clean_title(raw: Any, fallback: str) -> str:
    t = re.sub(r"<[^>]*>", "", str(raw or ""))
    t = re.sub(r"#\w+|[<>]", "", t)
    t = re.sub(r"\s+", " ", t).strip().strip("\"'\u201c\u201d").strip()
    if not t:
        t = re.sub(r"[<>]", "", str(fallback or "")).strip() or "Short clip"
    if t.isupper() and len(t) > 3:
        t = t[0] + t[1:].lower()
    if len(t) > 80:
        t = (t[:80].rsplit(" ", 1)[0].rstrip(",;:-\u2013 ") or t[:80])
    return t


def clean_hashtags(items: Any, limit: int = 4) -> List[str]:
    out: List[str] = []
    if not isinstance(items, (list, tuple)):
        return out
    for item in items:
        tag = re.sub(r"[\W_]+", "", str(item), flags=re.UNICODE)
        if len(tag) < 2 or len(tag) > 30 or not re.search(r"[^\W\d_]", tag):
            continue
        if tag.lower() in GENERIC_HASHTAGS or tag.lower() in {x.lower() for x in out}:
            continue
        out.append(tag)
        if len(out) >= limit:
            break
    return out


def clean_tags(items: Any, extra: Sequence[str] = ()) -> List[str]:
    out: List[str] = []
    seen = set()
    total = 0
    pool = list(items) if isinstance(items, (list, tuple)) else []
    for item in pool + list(extra):
        tag = re.sub(r"\s+", " ", re.sub(r"<[^>]*>|[<>#]", "", str(item))).strip()
        if not tag or len(tag) > 30 or tag.lower() in seen:
            continue
        if total + len(tag) + 1 > 450 or len(out) >= 12:
            break
        seen.add(tag.lower())
        out.append(tag)
        total += len(tag) + 1
    return out


def top_keywords(text: str, n: int = 3) -> List[str]:
    from collections import Counter

    words = re.findall(r"[^\W\d_]{5,}", text.lower())
    counts = Counter(w for w in words if w not in KEYWORD_STOPWORDS)
    return [w for w, _ in counts.most_common(n)]


def fallback_metadata(source_title: str, clip: Dict[str, Any], text: str) -> Dict[str, Any]:
    flat = re.sub(r"\s+", " ", text).strip()
    summary = flat[:200]
    if len(flat) > 200:
        summary = summary.rsplit(" ", 1)[0].rstrip(",;:") + "..."
    working = str(clip.get("title") or "").strip()
    if not working or re.match(r"^(clip \d+|clip)$", working, re.I):
        working = flat[:60].rsplit(" ", 1)[0] if len(flat) > 60 else flat
        words = working.split()
        while len(words) > 3 and len(words[-1].strip(",.;:!?")) <= 3:
            words.pop()
        working = " ".join(words)
    hashtags = [w.capitalize() for w in top_keywords(text, 3)]
    return {"title": working or source_title, "summary": summary, "hashtags": hashtags}


def generate_youtube_metadata(source_title: str, clip: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Ask the AI for a title, description and hashtags, then sanitise the result so
    YouTube will accept it. Falls back to something simple if the AI is unavailable."""
    text = str(clip.get("text") or "")
    data: Dict[str, Any] = {}
    if text.strip():
        prompt = build_metadata_prompt(source_title, str(clip.get("title") or ""), text)
        try:
            data = parse_json_object(ask_model(prompt, args.provider, args.ollama_model))
        except (Exception, SystemExit) as e:  # never let metadata break rendering or uploading
            print(f"    (metadata AI unavailable: {e}; using a simple fallback)")
    fb = fallback_metadata(source_title, clip, text)

    title = clean_title(data.get("title"), fb["title"])
    summary = str(data.get("description") or "")
    summary = re.sub(r"<[^>]*>", "", summary)
    summary = re.sub(r"https?://\S+|#\w+|[<>]", "", summary)
    summary = re.sub(r"\s+", " ", summary).strip()[:300] or fb["summary"]
    hashtags = clean_hashtags(data.get("hashtags"), limit=4) or fb["hashtags"]
    tags = clean_tags(data.get("tags"), extra=hashtags + ["Shorts"])

    parts = [summary] if summary else []
    source = f"From: {re.sub(r'[<>]', '', source_title)}"
    if clip.get("source_url"):
        source += f"\n{clip['source_url']}"
    parts.append(source)
    parts.append(" ".join(f"#{h}" for h in hashtags + ["Shorts"]))
    description = "\n\n".join(parts)[:4900]
    return {"title": title, "description": description, "tags": tags, "hashtags": hashtags}


def format_metadata_note(clip_name: str, meta: Dict[str, Any]) -> str:
    return (
        f"{clip_name}.mp4\n"
        f"TITLE: {meta['title']}\n"
        f"DESCRIPTION:\n{meta['description']}\n"
        f"TAGS: {', '.join(meta['tags'])}\n"
    )


def http_json_post(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json; charset=UTF-8"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def http_form_post(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def get_youtube_service(client_secrets: Path, token_file: Path):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "YouTube upload needs google-api-python-client, google-auth-oauthlib and google-auth. \n"
            "Install them with: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
        ) from exc

    creds = None
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), YOUTUBE_SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secrets.exists():
                raise RuntimeError(f"YouTube client secrets file not found: {client_secrets}")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)
        write_text_safe(token_file, creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path: Path, meta: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from googleapiclient.http import MediaFileUpload

    service = get_youtube_service(Path(args.youtube_client_secrets), Path(args.youtube_token_file))
    body = {
        "snippet": {
            "title": meta["title"],
            "description": meta["description"],
            "tags": meta.get("tags", []),
            "categoryId": args.youtube_category_id,
        },
        "status": {
            "privacyStatus": args.youtube_privacy,
            "selfDeclaredMadeForKids": args.youtube_made_for_kids,
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=10 * 1024 * 1024, resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"    YouTube upload: {status.progress() * 100:.1f}%")
    return response


def tiktok_init_upload(video_path: Path, meta: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    token = args.tiktok_access_token
    if not token:
        raise RuntimeError("TikTok upload needs --tiktok-access-token or TIKTOK_ACCESS_TOKEN.")

    size = video_path.stat().st_size
    chunk_size = max(1, int(args.tiktok_chunk_size))
    total_chunk_count = math.ceil(size / chunk_size)
    payload = {
        "post_info": {
            "title": meta["title"],
            "privacy_level": args.tiktok_privacy,
            "disable_duet": args.tiktok_disable_duet,
            "disable_comment": args.tiktok_disable_comment,
            "disable_stitch": args.tiktok_disable_stitch,
            "video_cover_timestamp_ms": args.tiktok_cover_ms,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunk_count,
        },
    }
    url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    headers = {"Authorization": f"Bearer {token}"}
    resp = http_json_post(url, payload, headers=headers)
    data = resp.get("data") or {}
    if (resp.get("error") or {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok init failed: {resp}")
    if not data.get("upload_url") or not data.get("publish_id"):
        raise RuntimeError(f"TikTok init returned an unexpected response: {resp}")
    return resp


def tiktok_upload_file(upload_url: str, video_path: Path, chunk_size: int) -> None:
    size = video_path.stat().st_size
    chunk_size = max(1, int(chunk_size))
    sent = 0
    with video_path.open("rb") as f:
        while sent < size:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            start = sent
            end = sent + len(chunk) - 1
            headers = {
                "Content-Type": "video/mp4",
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {start}-{end}/{size}",
            }
            req = urllib.request.Request(upload_url, data=chunk, headers=headers, method="PUT")
            with urllib.request.urlopen(req, timeout=1800) as resp:
                resp.read()
            sent += len(chunk)
            print(f"    TikTok upload: {min(100.0, sent * 100 / size):.1f}%")


def tiktok_fetch_status(publish_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    token = args.tiktok_access_token
    if not token:
        raise RuntimeError("TikTok upload needs --tiktok-access-token or TIKTOK_ACCESS_TOKEN.")
    url = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    payload = {"publish_id": publish_id}
    headers = {"Authorization": f"Bearer {token}"}
    return http_json_post(url, payload, headers=headers)


def upload_to_tiktok(video_path: Path, meta: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    init_resp = tiktok_init_upload(video_path, meta, args)
    data = init_resp["data"]
    upload_url = data["upload_url"]
    publish_id = data["publish_id"]
    tiktok_upload_file(upload_url, video_path, args.tiktok_chunk_size)

    # Best-effort status polling.
    status_resp = None
    for _ in range(int(args.tiktok_status_polls)):
        try:
            status_resp = tiktok_fetch_status(publish_id, args)
            break
        except Exception:
            time.sleep(float(args.tiktok_status_sleep))
    return {"init": init_resp, "status": status_resp, "publish_id": publish_id}


def instagram_publish_reel_from_url(video_url: str, caption: str, args: argparse.Namespace) -> Dict[str, Any]:
    token = args.instagram_access_token
    user_id = args.instagram_user_id
    if not token or not user_id:
        raise RuntimeError("Instagram upload needs --instagram-access-token and --instagram-user-id.")
    if not video_url:
        raise RuntimeError(
            "Instagram publishing needs --instagram-video-url. Meta's publishing API creates reels from a public video URL, not a local file."
        )

    api_version = args.instagram_api_version
    base = f"https://graph.facebook.com/{api_version}/{user_id}"
    create_payload = {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "share_to_feed": "true" if args.instagram_share_to_feed else "false",
        "access_token": token,
    }
    create_resp = http_form_post(f"{base}/media", create_payload)
    creation_id = create_resp.get("id") or create_resp.get("creation_id")
    if not creation_id:
        raise RuntimeError(f"Instagram container creation returned an unexpected response: {create_resp}")

    publish_resp = http_form_post(
        f"{base}/media_publish",
        {"creation_id": creation_id, "access_token": token},
    )
    return {"create": create_resp, "publish": publish_resp, "creation_id": creation_id}


def maybe_upload_clip(
    clip_path: Path,
    clip_meta: Dict[str, Any],
    source_title: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    targets = normalize_upload_targets(args.upload)
    if not targets:
        return {}

    results: Dict[str, Any] = {}
    if "youtube" in targets:
        yt_meta = clip_meta.get("youtube_meta") or clip_upload_metadata(source_title, clip_meta, "youtube")
        print("  Uploading to YouTube...")
        results["youtube"] = upload_to_youtube(clip_path, yt_meta, args)
        uploaded_id = results["youtube"].get("id") if isinstance(results["youtube"], dict) else None
        if uploaded_id:
            print(f"  ✓ YouTube upload finished: https://youtu.be/{uploaded_id}")
        else:
            print("  ✓ YouTube upload finished")
        yt_resp = results["youtube"] if isinstance(results["youtube"], dict) else {}
        actual = (yt_resp.get("status") or {}).get("privacyStatus")
        if actual:
            if actual != args.youtube_privacy:
                print(f"  ! You asked for {args.youtube_privacy}, but YouTube set this video to {actual}.")
                print("    Google restricts videos uploaded through the API by projects that haven't passed")
                print("    its audit. Check the video in YouTube Studio.")
            else:
                print(f"  Visibility: {actual}")

    if "tiktok" in targets:
        tt_meta = clip_upload_metadata(source_title, clip_meta, "tiktok")
        print("  Uploading to TikTok...")
        results["tiktok"] = upload_to_tiktok(clip_path, tt_meta, args)
        print("  ✓ TikTok upload finished")

    if "instagram" in targets:
        if not args.instagram_video_url:
            print("  Instagram skipped: provide --instagram-video-url (Instagram needs a public video URL).")
        else:
            ig_meta = clip_upload_metadata(source_title, clip_meta, "instagram")
            print("  Uploading to Instagram...")
            results["instagram"] = instagram_publish_reel_from_url(
                args.instagram_video_url,
                ig_meta["caption"],
                args,
            )
            print("  ✓ Instagram upload finished")

    return results


# --------------------------------------------------------------------------
# Per-video processing
# --------------------------------------------------------------------------
def process_video(url: str, args: argparse.Namespace, out_root: Path, index: int, total: int) -> None:
    print(f"\n=== Video {index}/{total} ===")
    info = probe_info(url)
    title = info.get("title", "video")
    video_id = info.get("id", f"video_{index}")

    # Same folder every run, so re-running a video reuses the download and transcript.
    video_root = out_root / safe_name(f"{title}_{video_id}")
    clips_dir = video_root / "clips"
    work_dir = out_root / "_work" / video_id  # downloads, caches, captions, AI replies
    ensure_dir(clips_dir)
    ensure_dir(work_dir)

    print(f"Output folder: {video_root}")

    print("1/4 Downloading video...")
    video, title, info = download(url, work_dir)
    print(f"  {title}")

    args.video_path = video

    print("2/4 Transcribing (the first run downloads the Whisper model)...")
    segments = transcribe(video, args.whisper)
    write_readable_transcript(segments, video_root / "transcript.txt")
    print(f"  Readable transcript saved to: {video_root / 'transcript.txt'}")

    print("3/4 Choosing clips...")
    clips = pick_clips(segments, args, work_dir)
    if not clips:
        print("  No usable clips were chosen.")
        return

    write_text_safe(video_root / "chosen_clips.json", json.dumps(clips, indent=2))

    # Remove numbered clips from an earlier run of this video so the folder stays tidy.
    for old in clips_dir.glob("*.mp4"):
        if re.match(r"^\d\d_.*\.mp4$", old.name):
            old.unlink()

    print(f"4/4 Rendering {len(clips)} clip(s)...")
    uploads_dir = video_root / "uploads"
    if normalize_upload_targets(args.upload):
        ensure_dir(uploads_dir)

    want_meta = "youtube" in normalize_upload_targets(args.upload) or getattr(args, "metadata", False)
    meta_notes: List[str] = []

    for i, c in enumerate(clips, 1):
        name = f"{i:02d}_{safe_name(str(c['title']))}"
        ass_path = work_dir / f"{name}.ass"
        out_path = clips_dir / f"{name}.mp4"
        build_subtitles(segments, c["start"], c["end"], ass_path)
        print(f"  Clip {i}: {c['title']} ({c['end'] - c['start']:.0f}s) - {c['reason']}")
        render(video, c["start"], c["end"], ass_path.name, str(out_path), work_dir)

        c["text"] = clip_transcript(segments, c["start"], c["end"])
        c["source_url"] = url
        if want_meta:
            print("  Writing YouTube title, description and hashtags...")
            c["youtube_meta"] = generate_youtube_metadata(title, c, args)
            print(f"    Title: {c['youtube_meta']['title']}")
            meta_notes.append(format_metadata_note(name, c["youtube_meta"]))

        if normalize_upload_targets(args.upload):
            clip_results = maybe_upload_clip(out_path, c, title, args)
            if clip_results:
                write_text_safe(uploads_dir / f"{name}.json", json.dumps(clip_results, indent=2))

    if meta_notes:
        write_text_safe(video_root / "metadata.txt", "\n----\n\n".join(meta_notes))
        print(f"Titles, descriptions and hashtags saved to: {video_root / 'metadata.txt'}")

    print(f"Done. Your clips are in: {clips_dir}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def expand_prompted_url_arguments(argv: Sequence[str]) -> List[str]:
    """Allow `URL --upload youtube` when a .command launcher reads the whole line
    as one string and passes it to Python as one positional argument.

    Normal shell/terminal invocations are left unchanged.
    """
    expanded: List[str] = []
    for arg in argv:
        value = str(arg)
        if value.startswith(("http://", "https://")) and " --" in value:
            try:
                expanded.extend(shlex.split(value))
            except ValueError:
                expanded.append(value)
        else:
            expanded.append(value)
    return expanded


# --------------------------------------------------------------------------
# Guided (interactive) mode: run with no arguments and answer simple questions
# --------------------------------------------------------------------------
def ask_yes_no(question: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"{question} [{hint}]: ").strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please type y or n.")


def looks_like_url(text: str) -> bool:
    t = text.strip().lower()
    return bool(t) and " " not in t and (t.startswith(("http://", "https://", "www.")) or "youtu" in t)


def clean_url(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith(("http://", "https://")) else "https://" + t


def youtube_upload_problem(args: argparse.Namespace) -> Optional[str]:
    """None if YouTube upload is ready to use, otherwise a short explanation of what's missing."""
    secrets = Path(args.youtube_client_secrets).expanduser()
    if not secrets.is_file():
        return f"client_secret.json wasn't found (looked here: {secrets})."
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        return ("the Google libraries aren't installed. Run: python -m pip install "
                "google-api-python-client google-auth-oauthlib google-auth-httplib2")
    return None


def ask_privacy(default: str) -> str:
    options = ["private", "unlisted", "public"]
    print("  Who should be able to see the uploads?")
    print("    1) private   - only you (recommended: check them first)")
    print("    2) unlisted  - anyone with the link")
    print("    3) public    - everyone")
    while True:
        try:
            raw = input(f"  Choose 1, 2 or 3 [{options.index(default) + 1}]: ").strip().lower()
        except EOFError:
            return default
        if not raw:
            choice = default
        elif raw in ("1", "2", "3"):
            choice = options[int(raw) - 1]
        elif raw in options:
            choice = raw
        else:
            print("  Please type 1, 2 or 3.")
            continue
        if choice == "public":
            print("  Note: YouTube forces API uploads from projects it hasn't audited to private,")
            print("  so this may still end up private.")
            if not ask_yes_no("  Go public straight away, with AI-written titles and no review?"):
                continue
        return choice


def interactive_queue(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
    """Ask for videos one at a time, with a yes/no for uploading each.
    Returns [(url, visibility or None)]; None means "make clips only, don't upload"."""
    print("\n=== Shorts Maker ===")
    print("Add your videos one at a time. Nothing starts until you've added them all.\n")
    queue: List[Tuple[str, Optional[str]]] = []
    youtube_problem: Optional[str] = "unchecked"
    try:
        while True:
            while True:
                raw = input(f"Video {len(queue) + 1} - paste the YouTube link: ").strip()
                if looks_like_url(raw):
                    break
                print("  That doesn't look like a link. Try again (Ctrl+C to quit).")
            url = clean_url(raw)

            privacy: Optional[str] = None
            if ask_yes_no("  Upload this video's clips to YouTube when they're ready?"):
                if youtube_problem == "unchecked":
                    youtube_problem = youtube_upload_problem(args)
                if youtube_problem:
                    print(f"  YouTube upload isn't set up: {youtube_problem}")
                    print("  I'll make the clips for this video without uploading.")
                else:
                    privacy = ask_privacy(args.youtube_privacy)
            queue.append((url, privacy))

            if not ask_yes_no("Add another video?"):
                break
            print()
    except EOFError:
        pass
    if not queue:
        sys.exit("No videos given.")

    print("\nHere's the plan:")
    for i, (url, privacy) in enumerate(queue, 1):
        what = f"make clips + upload to YouTube ({privacy})" if privacy else "make clips only"
        print(f"  {i}. {url}  ->  {what}")
    print()
    return queue


def main() -> None:
    p = argparse.ArgumentParser(description="YouTube video -> subtitled vertical shorts")
    p.add_argument("urls", nargs="*", help="YouTube video URLs (leave empty for guided mode)")
    p.add_argument("--clips", type=int, default=3, help="how many clips to make per video")
    p.add_argument("--min", type=int, default=30, help="minimum clip length (seconds)")
    p.add_argument("--max", type=int, default=60, help="maximum clip length (seconds)")
    p.add_argument(
        "--whisper",
        default="small",
        help="Whisper model: tiny, base, small, medium, large-v3 (bigger = more accurate, slower)",
    )
    p.add_argument("--out", default="shorts_output", help="output folder")
    p.add_argument(
        "--provider",
        choices=["claude", "ollama"],
        default=None,
        help="who picks the clips: claude (API key needed) or ollama (free, local). "
             "Default: claude if ANTHROPIC_API_KEY is set, otherwise ollama",
    )
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_TEXT_MODEL, help="model name for --provider ollama")
    p.add_argument(
        "--vision-model",
        default=DEFAULT_OLLAMA_VISION_MODEL,
        help="local Ollama vision model for frame analysis",
    )
    p.add_argument(
        "--use-vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="turn on sampled-frame vision analysis (slow; needs: ollama pull llava:7b)",
    )
    p.add_argument(
        "--vision-sample-seconds",
        type=int,
        default=8,
        help="sample one frame every N seconds for vision analysis",
    )
    p.add_argument(
        "--vision-batch-size",
        type=int,
        default=4,
        help="how many sampled frames to send to the vision model per request",
    )
    p.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write a YouTube title, description and hashtags for each clip to metadata.txt "
             "(done automatically with --upload youtube)",
    )
    p.add_argument(
        "--two-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rank evenly spaced candidate windows instead of letting the AI pick moments directly",
    )
    p.add_argument(
        "--clip",
        action="append",
        metavar="START-END",
        help="pick a clip yourself instead of using AI, e.g. --clip 1:20-2:05 (repeat for more clips)",
    )
    p.add_argument(
        "--upload",
        nargs="*",
        default=[],
        help="upload rendered clips to one or more platforms: youtube, tiktok, instagram, or all",
    )
    p.add_argument(
        "--youtube-client-secrets",
        default=str(SCRIPT_DIR / "client_secret.json"),
        help="Google OAuth client secrets JSON for YouTube uploads",
    )
    p.add_argument(
        "--youtube-token-file",
        default=str(SCRIPT_DIR / "youtube_token.json"),
        help="cached YouTube OAuth token file",
    )
    p.add_argument(
        "--youtube-privacy",
        choices=["private", "unlisted", "public"],
        default="private",
        help="YouTube privacy status for uploads",
    )
    p.add_argument(
        "--youtube-category-id",
        default="22",
        help="YouTube category id (default 22 = People & Blogs)",
    )
    p.add_argument(
        "--youtube-made-for-kids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="set the YouTube 'made for kids' flag",
    )
    p.add_argument(
        "--tiktok-access-token",
        default=os.environ.get("TIKTOK_ACCESS_TOKEN"),
        help="TikTok OAuth access token (or set TIKTOK_ACCESS_TOKEN)",
    )
    p.add_argument(
        "--tiktok-privacy",
        choices=["SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS", "PUBLIC_TO_EVERYONE"],
        default="SELF_ONLY",
        help="TikTok privacy level",
    )
    p.add_argument(
        "--tiktok-disable-comment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable TikTok comments for uploads",
    )
    p.add_argument(
        "--tiktok-disable-duet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable TikTok duet for uploads",
    )
    p.add_argument(
        "--tiktok-disable-stitch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable TikTok stitch for uploads",
    )
    p.add_argument(
        "--tiktok-cover-ms",
        type=int,
        default=1000,
        help="TikTok cover timestamp in milliseconds",
    )
    p.add_argument(
        "--tiktok-chunk-size",
        type=int,
        default=10_000_000,
        help="TikTok upload chunk size in bytes",
    )
    p.add_argument(
        "--tiktok-status-polls",
        type=int,
        default=3,
        help="how many times to poll TikTok status after upload",
    )
    p.add_argument(
        "--tiktok-status-sleep",
        type=float,
        default=5.0,
        help="seconds to wait between TikTok status polls",
    )
    p.add_argument(
        "--instagram-user-id",
        default=os.environ.get("INSTAGRAM_USER_ID"),
        help="Instagram professional account id (or set INSTAGRAM_USER_ID)",
    )
    p.add_argument(
        "--instagram-access-token",
        default=os.environ.get("INSTAGRAM_ACCESS_TOKEN"),
        help="Instagram/Facebook access token (or set INSTAGRAM_ACCESS_TOKEN)",
    )
    p.add_argument(
        "--instagram-video-url",
        default=None,
        help="public video URL used for Instagram Reel publishing",
    )
    p.add_argument(
        "--instagram-api-version",
        default="v27.0",
        help="Meta Graph API version for Instagram publishing",
    )
    p.add_argument(
        "--instagram-share-to-feed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share Instagram Reel to feed too",
    )

    args = p.parse_args(expand_prompted_url_arguments(sys.argv[1:]))
    args.youtube_client_secrets = str(Path(args.youtube_client_secrets).expanduser())
    args.youtube_token_file = str(Path(args.youtube_token_file).expanduser())

    queue: Optional[List[Tuple[str, Optional[str]]]] = None
    if not args.urls:
        if not sys.stdin.isatty():
            p.error("give at least one YouTube link, or run with no arguments for guided mode")
        queue = interactive_queue(args)
        args.urls = [u for u, _ in queue]

    if args.clip and len(args.urls) > 1:
        p.error("--clip only works with one link at a time")
    upload_targets = normalize_upload_targets(args.upload)
    if upload_targets:
        print("Upload enabled for: " + ", ".join(upload_targets))
        if "youtube" in upload_targets:
            client_secrets_path = Path(args.youtube_client_secrets).expanduser()
            args.youtube_client_secrets = str(client_secrets_path)
            args.youtube_token_file = str(Path(args.youtube_token_file).expanduser())
            print(f"  YouTube OAuth file: {args.youtube_client_secrets}")
            print(f"  YouTube token cache: {args.youtube_token_file}")
            if not client_secrets_path.is_file():
                sys.exit(
                    "YouTube upload was requested, but the OAuth file was not found. "
                    f"Put client_secret.json beside this Python script or pass "
                    f"--youtube-client-secrets. Checked: {client_secrets_path}"
                )
            try:
                import googleapiclient  # noqa: F401
                import google_auth_oauthlib  # noqa: F401
            except ImportError:
                sys.exit(
                    "YouTube upload dependencies are missing. Run: python3 -m pip install "
                    "--upgrade google-api-python-client google-auth-oauthlib google-auth-httplib2"
                )
    if args.provider is None:
        args.provider = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"
    check_setup(need_key=(not args.clip and args.provider == "claude"))

    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    failures = 0
    for i, url in enumerate(args.urls, 1):
        item_args = args
        if queue is not None:  # guided mode: each video has its own upload choice
            item_args = copy.copy(args)
            privacy = queue[i - 1][1]
            item_args.upload = ["youtube"] if privacy else []
            if privacy:
                item_args.youtube_privacy = privacy
        try:
            process_video(url, item_args, out_root, i, len(args.urls))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failures += 1
            print(f"\nERROR while processing {url}:\n{e}", file=sys.stderr)

    if failures:
        sys.exit(f"\nFinished with {failures} failed video(s).")
    print("\nAll videos processed successfully.")
    print(f"Your clips are in: {out_root}")


if __name__ == "__main__":
    main()
