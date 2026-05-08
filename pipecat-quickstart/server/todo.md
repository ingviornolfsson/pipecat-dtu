
## Subprocess-Level State Logging

### Motivation

The current logger captures high-level pipeline events (bot speaking, user turn, etc.) but not the internal states of individual services. Logging per-service state transitions would allow:

- Reconstructing high-level states (`bot_speaking`, `user_interrupting`, `processing`) from first principles rather than relying on the coarse `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` heuristic.
- Debugging latency (e.g. how much time is LLM TTFB vs. TTS startup vs. audio buffering).
- Detecting failure modes like TTS stall, LLM timeout, or STT silence.

### States to capture per service

| Service | Key frames / events |
|---|---|
| STT (Deepgram) | `TranscriptionFrame` (final), `InterimTranscriptionFrame`, VAD passthrough |
| LLM (OpenAI) | `LLMFullResponseStartFrame`, `LLMFullResponseEndFrame`, `LLMTextFrame` (token stream), `LLMRunFrame` (trigger) |
| TTS (Cartesia) | `TTSStartedFrame`, `TTSStoppedFrame`, `TTSTextFrame`, `TTSAudioRawFrame` (first chunk = startup latency) |
| Transport output | `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `BotSpeakingFrame` |
| VAD / Turn | `VADUserStartedSpeakingFrame`, `VADUserStoppedSpeakingFrame`, `UserStoppedSpeakingFrame` |
| Interruption | `InterruptionFrame` (marks the moment the bot was cut off) |

### Translation to high-level states

A simple FSM reconstructed from subprocess events:

- `idle` → `listening`: after `BotStoppedSpeakingFrame` (or on session start)
- `listening` → `user_speaking`: `VADUserStartedSpeakingFrame`
- `user_speaking` → `processing`: `UserStoppedSpeakingFrame` + `LLMRunFrame`
- `processing` → `speaking`: `BotStartedSpeakingFrame` (with sub-phases: LLM streaming → TTS startup → audio playback)
- `speaking` → `user_interrupting`: `VADUserStartedSpeakingFrame` while `bot_speech = active`; confirm with `InterruptionFrame`
- `user_interrupting` → `user_speaking`: `InterruptionFrame` clears the bot queue; state fully transitions on `BotStoppedSpeakingFrame`

### Implementation approach

Extend `ConversationLogObserver.on_push_frame` to emit fine-grained `service_state_changed` events alongside the existing coarse events. Each event would carry `service` (e.g. `"tts"`, `"llm"`), `state`, and `ts_ns`. The high-level state reconstruction can then be done as a post-processing step on the JSONL log rather than in the hot path.

# Plan: Long-Pause User Turn Bugs (for todo.md)

Insert the following section at the TOP of todo.md (before "## Subprocess-Level State Logging"):

---

## Long-Pause User Turn: Bugs and Fixes

### Background

When a user pauses mid-sentence and then resumes (e.g., "I've been thinking... about what to see."), several interacting bugs in `ConversationLogObserver` and `PeriodicSmartTurnAnalyzer` cause incorrect or incomplete log entries. Confirmed in `logs/session_20260507_164433_4ca138ab.jsonl`, turn 5 (t=38.7–45.3s).

### Observed symptoms

1. **Duplicate `user_turn` events** — Two entries with the same `start_ts_ns`. First has pre-pause transcript ("I've been thinking"), second has empty text ~780ms later.
2. **Transcript truncated at pause** — Post-pause speech ("about what to see.") silently dropped.
3. **Turn probability sampling gap** — Periodic samples stop at the pause and do not resume (~2.6s gap), even though the turn is still active.
4. **VAD sub-events invisible** — `VADUserStoppedSpeakingFrame` and the subsequent `VADUserStartedSpeakingFrame` within a turn are never logged; visualiser cannot show mid-turn pauses.

### Root causes

#### Bug A — Duplicate `user_turn` (ConversationLogObserver)

`UserStoppedSpeakingFrame` fires twice: once for the pause, once for the resumed segment.
Observer logs `user_turn` unconditionally on each. On the second firing, `_in_user_turn=False`
and `_user_transcript_parts=[]`, so text="" and `_user_turn_start_ts` retains the original value.

**Fix**: Guard the `UserStoppedSpeakingFrame` branch with `if self._in_user_turn:`. One-liner in `on_push_frame`.

#### Bug B — Post-pause transcript dropped (ConversationLogObserver)

After the first `UserStoppedSpeakingFrame`, `_in_user_turn=False`. Any `TranscriptionFrame`
for the resumed speech hits `if self._in_user_turn:` and is dropped.

**Fix B1 (recommended)**: Apply Fix A. The resumed speech will produce its own complete
`user_turn` entry (new `start_ts_ns` from the second `VADUserStartedSpeakingFrame`). Both
segments logged separately; pause is implicit from the gap. No extra state needed.

**Fix B2 (complex, defer)**: Keep `_in_user_turn=True` after first `UserStoppedSpeakingFrame`
if a second VAD onset arrives within ~500ms. Only finalize on a UserStopped not followed by
another onset. Mirrors turn strategy logic. Not recommended until Fix B1 is validated.

#### Bug C — Periodic sampling stops at pause (PeriodicSmartTurnAnalyzer)

`append_audio()` gates on `self._speech_triggered` (parent class field). During pause,
`is_speech=False` causes parent to clear `_speech_triggered`. Sampling stays off for
the resumed speech.

**Fix**: Track onset independently in `PeriodicSmartTurnAnalyzer`:
- Add `self._own_speech_active: bool = False` in `__init__`
- In `append_audio`: set `self._own_speech_active = True` when `is_speech=True`
- Gate periodic sampling on `self._own_speech_active` instead of `self._speech_triggered`
- In `clear()`: reset `self._own_speech_active = False`

#### Enhancement D — Log mid-turn VAD sub-events (ConversationLogObserver)

Add unconditional `vad_activity_changed` events on `VADUserStopped/StartedSpeakingFrame`
(regardless of `_in_user_turn` state). Payload: `{"active": bool}`.

In `plot_session.py`, add a VAD activity overlay or 4th subplot (filled boolean step)
so pause gaps in turn_probability are visually explained.

### Recommended fix order

1. **Fix A** — 1 line, eliminates duplicate `user_turn` noise. Quick win.
2. **Fix C** — ~5 lines in PeriodicSmartTurnAnalyzer, eliminates sampling gap.
3. **Enhancement D** — ~10 lines in observer + plot update, adds VAD visibility.
4. **Fix B1** — falls out naturally after Fix A; verify in a new session log.
5. **Fix B2** — only if merging segments is needed for downstream analysis.
