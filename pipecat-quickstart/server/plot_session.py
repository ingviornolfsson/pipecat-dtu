#!/usr/bin/env python3
"""Plot VAD/Turn activity and turn probability in a single panel.

Usage::

    python plot_vad.py [logfile | index]
    python plot_vad.py --save plot.png
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.patches as mpatches  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Loading & deduplication  (identical to plot_session.py)
# ---------------------------------------------------------------------------

def load_events(path: Path) -> tuple[list[dict], dict]:
    raw: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING line {lineno}: JSON parse error — {exc}", file=sys.stderr)

    seen_exact: dict[str, int] = {}
    exact_dup_count = 0
    deduped: list[dict] = []
    for ev in raw:
        key = json.dumps(ev, sort_keys=True)
        if key in seen_exact:
            exact_dup_count += 1
        else:
            seen_exact[key] = 1
            deduped.append(ev)

    for ev in deduped:
        ev["ts_s"] = ev["ts_ns"] / 1e9

    MAX_SESSION_S = 600.0
    outliers = [ev for ev in deduped if ev["ts_s"] > MAX_SESSION_S]
    deduped = [ev for ev in deduped if ev["ts_s"] <= MAX_SESSION_S]

    report = {
        "raw_count": len(raw),
        "exact_dup_count": exact_dup_count,
        "after_dedup": len(deduped),
        "outlier_count": len(outliers),
    }
    return deduped, report


def print_quality_report(report: dict, path: Path) -> None:
    raw = report["raw_count"]
    exact = report["exact_dup_count"]
    after = report["after_dedup"]
    outliers = report.get("outlier_count", 0)

    issues = exact > 0 or outliers > 0
    tag = "WARNING" if issues else "OK"

    print(f"\n=== Data-quality report for {path.name} [{tag}] ===")
    print(f"  Raw lines:            {raw}")
    print(f"  Exact duplicates:     {exact}  (removed)")
    print(f"  Events after dedup:   {after}")
    if outliers > 0:
        pct = 100 * outliers / raw
        print(
            f"\n  ⚠  {outliers} event(s) ({pct:.1f}% of raw) had ts_ns > 10 min "
            "and were removed."
        )
    print()


# ---------------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------------

def build_vad_series(events: list[dict]) -> tuple[list[float], list[int], list[float]]:
    times, active, turn_ends = [0.0], [0], []
    for e in events:
        if e["event"] == "vad_started":
            times.append(e["ts_s"]); active.append(1)
        elif e["event"] == "vad_stopped":
            times.append(e["ts_s"]); active.append(0)
        elif e["event"] == "user_turn_end":
            turn_ends.append(e["ts_s"])
    return times, active, turn_ends


def build_turn_prob_series(events: list[dict]) -> tuple[list[float], list[float], list[bool]]:
    tp = [e for e in events if e["event"] == "turn_probability"]
    return (
        [e["ts_s"] for e in tp],
        [e["probability"] for e in tp],
        [e.get("is_complete", False) for e in tp],
    )


def build_bot_panel_data(
    events: list[dict], t_max: float
) -> tuple[
    list[float], list[float], list[float],
    list[tuple[float, float]], list[tuple[float, float, str]]
]:
    """Return LLM start times, stop times, token times, bot-speaking spans,
    and bot utterances paired to bot-speaking spans.
    """
    llm_starts: list[float] = []
    llm_stops: list[float] = []
    llm_token_times: list[float] = []
    bot_spans: list[tuple[float, float]] = []
    bot_utterances: list[tuple[float, float, str]] = []

    llm_start: float | None = None
    llm_tokens: list[str] = []
    bot_start: float | None = None

    for e in sorted(events, key=lambda x: x["ts_s"]):
        ev = e["event"]
        if ev == "llm_started":
            llm_start = float(e["ts_s"])
            llm_starts.append(llm_start)
            llm_tokens = []
        elif ev == "llm_token" and llm_start is not None:
            llm_token_times.append(e["ts_s"])
            llm_tokens.append(e.get("text", ""))
        elif ev == "llm_stopped" and llm_start is not None:
            llm_stops.append(e["ts_s"])
            bot_utterances.append((llm_start, e["ts_s"], "".join(llm_tokens)))
            llm_start = None
            llm_tokens = []
        elif ev == "bot_speaking_started":
            bot_start = e["ts_s"]
        elif ev == "bot_speaking_stopped" and bot_start is not None:
            bot_spans.append((bot_start, e["ts_s"]))
            bot_start = None

    if llm_start is not None:
        llm_stops.append(t_max)
        bot_utterances.append((llm_start, t_max, "".join(llm_tokens)))
    if bot_start is not None:
        bot_spans.append((bot_start, t_max))

    # Interruption times — used to mark incomplete bot turns with "...".
    interruption_times = [e["ts_s"] for e in events if e["event"] == "interruption"]

    def _was_interrupted(span_end: float) -> bool:
        return any(abs(t - span_end) < 0.2 for t in interruption_times)

    # Pair each LLM utterance with the first bot-speaking span that starts
    # at or after the LLM span start, so text can be anchored to bot speech.
    paired: list[tuple[float, float, str]] = []
    for llm_s, llm_e, text in bot_utterances:
        match = next(
            ((bs, be) for bs, be in bot_spans if bs >= llm_s - 0.5),
            None,
        )
        end_time = match[1] if match else llm_e
        suffix = " ..." if _was_interrupted(end_time) else ""
        if match:
            paired.append((match[0], match[1], text + suffix))
        else:
            paired.append((llm_s, llm_e, text + suffix))

    # If tts_text events are present (new log format), use them for accurate
    # spoken text — words not played due to an interruption are never logged.
    tts_text_events = [e for e in events if e["event"] == "tts_text"]
    if tts_text_events and bot_spans:
        ctx_words: dict[str, list[str]] = {}
        for e in sorted(tts_text_events, key=lambda x: x["ts_s"]):
            cid = e.get("context_id") or ""
            ctx_words.setdefault(cid, []).append(e.get("text", ""))

        ctx_first_audio: dict[str, float] = {
            e["context_id"]: e["ts_s"]
            for e in events
            if e["event"] == "tts_first_audio" and e.get("context_id")
        }

        spoken_paired: list[tuple[float, float, str]] = []
        for cid, words in ctx_words.items():
            text = " ".join(w.strip() for w in words if w.strip())
            if not text:
                continue
            fa_time = ctx_first_audio.get(cid)
            if fa_time is not None:
                match = min(bot_spans, key=lambda s, t=fa_time: abs(s[0] - t))
                if abs(match[0] - fa_time) < 0.5:
                    suffix = " ..." if _was_interrupted(match[1]) else ""
                    spoken_paired.append((match[0], match[1], text + suffix))
        return llm_starts, llm_stops, llm_token_times, bot_spans, spoken_paired

    return llm_starts, llm_stops, llm_token_times, bot_spans, paired


def build_transcript_turns(events: list[dict]) -> list[tuple[float, float, float, str]]:
    """Group stt_final texts into turns bounded by vad_started / user_turn_end.

    Returns (vad_start, vad_stop, turn_end, text). turn_end is the user_turn_end
    timestamp, used as the right anchor for the transcript label.
    """
    turns: list[tuple[float, float, float, str]] = []
    current_start: float | None = None
    current_stop: float | None = None
    current_texts: list[str] = []
    for e in sorted(events, key=lambda x: x["ts_s"]):
        ev = e["event"]
        if ev == "vad_started":
            if current_start is None:
                # First VAD segment of a new turn — initialise everything.
                current_start = e["ts_s"]
                current_stop = None
                current_texts = []
            # else: mid-turn VAD restart (user paused briefly) — keep
            # accumulated texts and the original turn start; current_stop
            # will be updated again when vad_stopped fires.
        elif ev == "vad_stopped" and current_start is not None:
            current_stop = e["ts_s"]
        elif ev == "stt_final" and current_start is not None:
            current_texts.append(e.get("text", ""))
        elif ev == "user_turn_end" and current_start is not None:
            text = " ".join(current_texts).strip()
            stop = current_stop if current_stop is not None else e["ts_s"]
            if text:
                turns.append((current_start, stop, e["ts_s"], text))
            current_start = None
            current_stop = None
            current_texts = []
    return turns


def build_tts_gantt(events: list[dict], t_max: float) -> list[dict]:
    """Return per-context TTS bars with greedy row packing to avoid overlap."""
    starts: dict[str, float] = {}
    first_audio: dict[str, float] = {}
    results: list[dict] = []

    for e in sorted(events, key=lambda x: x["ts_s"]):
        cid = e.get("context_id") or ""
        if e["event"] == "tts_started":
            starts[cid] = e["ts_s"]
        elif e["event"] == "tts_first_audio" and cid:
            first_audio[cid] = e["ts_s"]
        elif e["event"] == "tts_stopped" and cid in starts:
            results.append({
                "context_id": cid,
                "start_s": starts.pop(cid),
                "end_s": e["ts_s"],
                "first_audio_s": first_audio.get(cid),
            })

    for cid, start in starts.items():
        results.append({
            "context_id": cid,
            "start_s": start,
            "end_s": t_max,
            "first_audio_s": first_audio.get(cid),
        })

    row_ends: list[float] = []
    for bar in sorted(results, key=lambda x: x["start_s"]):
        placed = False
        for r, end in enumerate(row_ends):
            if bar["start_s"] >= end - 1e-6:
                bar["row"] = r
                row_ends[r] = bar["end_s"]
                placed = True
                break
        if not placed:
            bar["row"] = len(row_ends)
            row_ends.append(bar["end_s"])

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Y-axis layout
# 0 = VAD off / turn_prob 0
# 1 = VAD on  / turn_prob 1  ← top of visible area; text hangs downward from here
Y_BOTTOM = -0.08
Y_TOP = 1.25
TEXT_MARGIN_S = 0.25  # seconds of padding between text anchor and span edge


def plot(events: list[dict], title: str, save_path: str | None = None) -> None:
    t_max = max((e["ts_s"] for e in events), default=1.0) * 1.05

    vad_times, vad_active, turn_ends = build_vad_series(events)
    tp_times, tp_probs, tp_complete = build_turn_prob_series(events)
    llm_starts, llm_stops, llm_token_times, bot_spans, bot_utterances = build_bot_panel_data(events, t_max)
    tts_bars = build_tts_gantt(events, t_max)
    stt_interim_times = [e["ts_s"] for e in events if e["event"] == "stt_interim"]
    stt_final_events = [e for e in events if e["event"] == "stt_final"]
    interruption_times = [e["ts_s"] for e in events if e["event"] == "interruption"]
    cmap = matplotlib.colormaps["tab10"]  # type: ignore[attr-defined]

    fig, (ax_vad, ax_bot) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )
    ax_tp = ax_vad.twinx()

    # =========================================================================
    # Panel 1 — VAD / Turn probability
    # =========================================================================

    # ylim set after tight_layout to prevent twinx from resetting it (see end of function)

    # VAD as axvspan patches (like bot panel)
    VAD_YMIN, VAD_YMAX = 0.05, 0.95
    vad_on: float | None = None
    for t, a in zip(vad_times, vad_active):
        if a == 1 and vad_on is None:
            vad_on = t
        elif a == 0 and vad_on is not None:
            ax_vad.axvspan(vad_on, t, ymin=VAD_YMIN, ymax=VAD_YMAX, color="#66bb6a", alpha=0.45, zorder=1)
            vad_on = None
    if vad_on is not None:
        ax_vad.axvspan(vad_on, t_max, ymin=VAD_YMIN, ymax=VAD_YMAX, color="#66bb6a", alpha=0.45, zorder=1)

    for t in stt_interim_times:
        ax_vad.axvline(t, ymin=VAD_YMIN, ymax=VAD_YMAX, color="#aaaaaa", linewidth=0.9, alpha=0.85, zorder=3)
    for t in turn_ends:
        ax_vad.axvline(t, color="#2e7d32", linestyle="--", linewidth=1.2, alpha=0.9, zorder=3)

    ax_vad.set_yticks([])
    ax_vad.set_ylabel("VAD active", fontsize=10)
    ax_vad.grid(axis="x", linestyle=":", alpha=0.4)
    ax_vad.set_xlim(0, t_max)

    ax_tp.set_yticks([0.0, 0.5, 1.0])
    ax_tp.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=9)
    ax_tp.set_ylabel("Turn probability", fontsize=10)

    if tp_times:
        # Break the connecting line wherever a user_turn_end falls between samples
        # so long cross-turn segments are not drawn.
        line_t: list[float] = [tp_times[0]]
        line_p: list[float] = [tp_probs[0]]
        for i in range(1, len(tp_times)):
            if any(tp_times[i - 1] <= te <= tp_times[i] for te in turn_ends):
                line_t.append(float("nan"))
                line_p.append(float("nan"))
            line_t.append(tp_times[i])
            line_p.append(tp_probs[i])
        ax_tp.plot(line_t, line_p, color="gray", linewidth=0.8, zorder=1)
        dot_colors = ["#e57373" if c else "#81c784" for c in tp_complete]
        ax_tp.scatter(tp_times, tp_probs, c=dot_colors, s=25, zorder=2)
        ax_tp.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)

    # stt_final textboxes — one box per event, centred at its timestamp.
    # Boxes alternate between two levels; texts longer than 25 chars are wrapped.
    # Level y positions are adjusted after first render so each level is tall
    # enough for its tallest box (see _adjust_stt_levels below).
    STT_LINE_BASE = 1.20  # base of the tick line (inside axes area)
    STT_Y_BASE = 1.27     # bottom of level 0

    stt_artists: list[tuple] = []
    for i, e in enumerate(stt_final_events):
        level = i % 2
        raw = e.get("text", "")
        text = textwrap.fill(raw, 25) if len(raw) > 25 else raw
        t = ax_tp.text(
            e["ts_s"], STT_Y_BASE, text,
            fontsize=7, ha="center", va="bottom",
            clip_on=False, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="#999999", alpha=0.85),
        )
        (tick,) = ax_tp.plot(
            [e["ts_s"], e["ts_s"]], [STT_LINE_BASE, STT_Y_BASE],
            color="black", linewidth=0.8, alpha=0.5, clip_on=False, zorder=4,
        )
        stt_artists.append((t, tick, level))

    vad_legend: list = [mpatches.Patch(color="#66bb6a", alpha=0.6, label="vad active")]
    if stt_interim_times:
        import matplotlib.lines as _mlines  # type: ignore[import-untyped]
        vad_legend.append(_mlines.Line2D([], [], color="#aaaaaa", linewidth=0.9, alpha=0.85, label="stt_interim"))
    if turn_ends:
        vad_legend.append(mpatches.Patch(color="#2e7d32", alpha=0.9, label="user_turn_end"))
    if tp_times:
        vad_legend += [
            mpatches.Patch(color="#81c784", label="turn_prob (incomplete)"),
            mpatches.Patch(color="#e57373", label="turn_prob (complete)"),
        ]
    if stt_final_events:
        vad_legend.append(mpatches.Patch(fc="lightyellow", ec="#999999", alpha=0.85, label="stt_final"))
    ax_vad.legend(handles=vad_legend, fontsize=8, loc="upper right")

    # =========================================================================
    # Panel 2 — LLM + Bot speech
    # =========================================================================

    # Both LLM and bot-speaking spans sit on the same y band [0, 1].
    # Text is placed inside the bot-speaking boxes (y centred at 0.5).
    BOT_Y_BOTTOM = -0.05
    BOT_Y_TOP = 1.55
    TTS_LINE_Y = 1.30

    # Normalised ymin/ymax for axvspan (axes fraction) — fill most of the axis
    BOT_SPAN_YMIN = 0.05
    BOT_SPAN_YMAX = 0.75  # leave headroom for TTS line
    BOT_TEXT_Y = 0.5   # data coords — centre of the bar
    # Centre of the bot-speaking axvspan band in data coordinates
    BOT_SPAN_CENTER_Y = BOT_Y_BOTTOM + (BOT_SPAN_YMIN + BOT_SPAN_YMAX) / 2 * (BOT_Y_TOP - BOT_Y_BOTTOM)

    # ylim set after tight_layout (see end of function)

    # LLM token events — thin vertical lines
    for t_tok in llm_token_times:
        ax_bot.axvline(t_tok, ymin=BOT_SPAN_YMIN, ymax=BOT_SPAN_YMAX,
                       color="#ffb74d", linewidth=0.6, alpha=0.6, zorder=1)
    # LLM start events — solid orange line
    for t_s in llm_starts:
        ax_bot.axvline(t_s, ymin=BOT_SPAN_YMIN, ymax=BOT_SPAN_YMAX,
                       color="#e65100", linewidth=1.5, alpha=0.9, zorder=2)
    # LLM stop events — dashed orange line
    for t_s in llm_stops:
        ax_bot.axvline(t_s, ymin=BOT_SPAN_YMIN, ymax=BOT_SPAN_YMAX,
                       color="#e65100", linewidth=1.5, linestyle="--", alpha=0.9, zorder=2)

    # Bot speaking spans (teal) — drawn on top of LLM lines
    for start, end in bot_spans:
        ax_bot.axvspan(start, end, ymin=BOT_SPAN_YMIN, ymax=BOT_SPAN_YMAX,
                       color="#4dd0e1", alpha=0.55, zorder=2)

    # Interruption events — red crosses centred on the TTS span line
    if interruption_times:
        ax_bot.plot(
            interruption_times, [TTS_LINE_Y] * len(interruption_times),
            marker="x", color="red", linestyle="none",
            markersize=9, markeredgewidth=2.0, zorder=6,
        )

    # TTS spans — thick colored lines above the bot-speaking fill
    for i, bar in enumerate(tts_bars):
        color = cmap(i % 10)
        ax_bot.plot([bar["start_s"], bar["end_s"]], [TTS_LINE_Y, TTS_LINE_Y],
                    color=color, linewidth=6, alpha=0.75, solid_capstyle="butt", zorder=3)
        if bar["first_audio_s"] is not None:
            ax_bot.plot(bar["first_audio_s"], TTS_LINE_Y, marker="|", color="black",
                        markersize=12, markeredgewidth=2.5, zorder=4)

    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Bot activity", fontsize=10)
    ax_bot.set_xlabel("Time (s)", fontsize=10)
    ax_bot.grid(axis="x", linestyle=":", alpha=0.4)

    # Bot utterance text — left-aligned to the start of the bot-speaking span
    bot_text_objects: list[tuple] = []
    for start, end, orig_text in bot_utterances:
        t = ax_bot.text(
            start + TEXT_MARGIN_S, BOT_TEXT_Y, "",
            fontsize=8, va="center", ha="left",
            clip_on=True, color="black", zorder=4,
        )
        bot_text_objects.append((t, start, end, orig_text))

    import matplotlib.lines as mlines  # type: ignore[import-untyped]
    bot_legend = [
        mlines.Line2D([], [], color="#ffb74d", linewidth=0.8, alpha=0.8, label="LLM token"),
        mlines.Line2D([], [], color="#e65100", linewidth=1.5, label="LLM start"),
        mlines.Line2D([], [], color="#e65100", linewidth=1.5, linestyle="--", label="LLM stop"),
        mpatches.Patch(color="#4dd0e1", alpha=0.65, label="bot speaking"),
    ]
    if interruption_times:
        bot_legend.append(
            mlines.Line2D([], [], marker="x", color="red", linestyle="none",
                          markersize=7, markeredgewidth=2.0, label="interruption")
        )
    if tts_bars:
        bot_legend.append(mpatches.Patch(color="#888888", alpha=0.75, label="tts span"))
    if any(b["first_audio_s"] is not None for b in tts_bars):
        bot_legend.append(
            mlines.Line2D([], [], marker="|", color="black", linestyle="none",
                          markersize=9, markeredgewidth=2.0, label="tts_first_audio")
        )
    ax_bot.legend(handles=bot_legend, fontsize=8, loc="upper right")

    # =========================================================================
    # Dynamic text wrapping (both panels share the same x geometry)
    # =========================================================================
    # Track the axes-width (px) used for the last wrap so any change triggers
    # a recompute, including Windows 11 snap/maximise which may skip resize_event.
    _state: dict = {"last_ax_px": -1.0, "updating": False}

    def _ax_width_px() -> float:
        return fig.get_figwidth() * fig.dpi * ax_vad.get_position().width

    def _char_px(fontsize: float) -> float:
        return fontsize * 0.60 * (fig.dpi / 72.0)

    def _update_wrapping(event=None) -> None:
        if _state["updating"]:
            return
        ax_px = _ax_width_px()
        if ax_px <= 1 or ax_px == _state["last_ax_px"]:
            return
        _state["updating"] = True
        x_min, x_max = ax_vad.get_xlim()
        px_per_data = ax_px / (x_max - x_min)
        char_px = _char_px(8)
        for t, start, end, orig_text in bot_text_objects:
            span_px = (end - start) * px_per_data
            chars_per_line = max(8, int(1 * span_px / char_px))
            t.set_text(textwrap.fill(orig_text, width=chars_per_line))
        _state["last_ax_px"] = ax_px
        _state["updating"] = False
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", _update_wrapping)
    fig.canvas.mpl_connect("resize_event", lambda e: _update_wrapping())
    try:
        # configure_event fires on Windows when the window is snapped/maximised
        fig.canvas.mpl_connect("configure_event", lambda e: _update_wrapping())
    except Exception:
        pass

    # Adjust STT level y positions after first render so each level is tall
    # enough for its tallest (possibly wrapped) box.
    _stt_done: dict = {"v": False}

    def _adjust_stt_levels(*_) -> None:
        if _stt_done["v"] or not stt_artists:
            return
        try:
            renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        except AttributeError:
            return
        pts = ax_tp.transData.transform([[0, 0], [0, 1]])
        px_per_data_y = abs(pts[1][1] - pts[0][1])
        if px_per_data_y < 1:
            return
        max_h: dict[int, float] = {0: 0.0, 1: 0.0}
        for t, tick, level in stt_artists:
            h = t.get_window_extent(renderer=renderer).height / px_per_data_y
            max_h[level] = max(max_h[level], h)
        y_level = {0: STT_Y_BASE, 1: STT_Y_BASE + max_h[0] + 0.02}
        for t, tick, level in stt_artists:
            y = y_level[level]
            t.set_position((t.get_position()[0], y))
            tick.set_ydata([STT_LINE_BASE, y])
        _stt_done["v"] = True
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("draw_event", _adjust_stt_levels)

    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    plt.tight_layout()

    # Set ylims after tight_layout so twinx cannot reset them
    ax_vad.set_ylim(Y_BOTTOM, Y_TOP)
    ax_tp.set_ylim(Y_BOTTOM, Y_TOP)
    ax_bot.set_ylim(BOT_Y_BOTTOM, BOT_Y_TOP)

    _update_wrapping()
    _adjust_stt_levels()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point  (identical argument parsing to plot_session.py)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VAD/Turn panel from a pipecat session JSONL log.")
    parser.add_argument("logfile", nargs="?", default=None, help="Path to the session .jsonl file")
    parser.add_argument("--save", metavar="PATH", help="Save plot to file instead of displaying")
    args = parser.parse_args()

    if args.logfile is None:
        log_idx: int | None = 0
    elif args.logfile.lstrip("-").isdigit():
        log_idx = -int(args.logfile)
    else:
        path = Path(args.logfile)
        log_idx = None

    if log_idx is not None:
        log_dir = Path("logs")
        if not log_dir.exists() or not log_dir.is_dir():
            print("ERROR: logs/ directory not found in current path", file=sys.stderr)
            sys.exit(1)
        log_files = sorted(
            log_dir.glob("session_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not log_files:
            print("ERROR: no session_*.jsonl files found in logs/", file=sys.stderr)
            sys.exit(1)
        path = log_files[log_idx]

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    events, report = load_events(path)
    print_quality_report(report, path)
    plot(events, title=f"Session: {path.stem}", save_path=args.save)


if __name__ == "__main__":
    main()
