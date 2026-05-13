
## Subprocess-Level State Logging — Two-Tier Architecture

**Tier 1 (raw event logging) must be implemented and validated before any work on Tier 2 begins.**

The current `ConversationLogObserver` implementation is closer to Tier 2 — it maintains FSM state inline and emits coarse, pre-interpreted events. This approach should be **thoroughly reworked or discarded** in favour of a Tier 1 foundation. High-level states are a post-processing concern, not a pipeline concern.

---

### Tier 1 — Raw Service Event Logging (implement first)

**Goal:** Record every meaningful frame boundary as a timestamped JSONL event, with zero interpretation. The log is the ground truth; all analysis is downstream.

**Key insight from codebase research:** Services have strictly paired start/end frames that are reliable logging primitives. States overlap *across* services by design (LLM streams while TTS synthesises while audio plays), so a single coarse state machine cannot be derived in-band without priority rules. See `docs/pipecat_lifecycle_frames.md` for full frame inventory.

#### Events to capture per service

| Service | Frame → event | Payload |
|---|---|---|
| VAD | `VADUserStartedSpeakingFrame` → `vad_started` | `ts_ns` |
| VAD | `VADUserStoppedSpeakingFrame` → `vad_stopped` | `ts_ns` |
| Turn | `UserStoppedSpeakingFrame` → `user_turn_end` | `ts_ns` |
| STT | `InterimTranscriptionFrame` → `stt_interim` | `ts_ns`, `text` |
| STT | `TranscriptionFrame` (final) → `stt_final` | `ts_ns`, `text` |
| LLM | `LLMFullResponseStartFrame` → `llm_started` | `ts_ns` |
| LLM | `LLMFullResponseEndFrame` → `llm_stopped` | `ts_ns` |
| LLM | `LLMTextFrame` → `llm_token` | `ts_ns`, `text` (optional; high volume) |
| LLM | `FunctionCallInProgressFrame` → `llm_function_call` | `ts_ns`, `function_name` |
| TTS | `TTSStartedFrame` → `tts_started` | `ts_ns`, `context_id` |
| TTS | `TTSStoppedFrame` → `tts_stopped` | `ts_ns`, `context_id` |
| TTS | `TTSAudioRawFrame` (first per context) → `tts_first_audio` | `ts_ns`, `context_id` |
| Transport | `BotStartedSpeakingFrame` → `bot_speaking_started` | `ts_ns` |
| Transport | `BotStoppedSpeakingFrame` → `bot_speaking_stopped` | `ts_ns` |
| Interruption | `InterruptionFrame` → `interruption` | `ts_ns` |

#### Frame pairing properties (from codebase research)

- **STT:** `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` — strict pair, guaranteed by VAD
- **LLM:** `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` — strict pair, `LLMFullResponseEndFrame` always emitted in `finally` block
- **TTS:** `TTSStartedFrame(context_id=X)` / `TTSStoppedFrame(context_id=X)` — strict pair per context; multiple contexts can interleave
- **Transport:** `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` — strict pair
- **InterruptionFrame:** no matching end frame; it is a point-in-time signal

#### Implementation

Replace (or gut) `ConversationLogObserver` with a minimal observer that:
1. Handles each frame type listed above in `on_push_frame`
2. Writes a single JSONL line per event: `{"ts_ns": <int>, "event": "<name>", ...payload}`
3. Maintains **no FSM state** — no `_bot_state`, no `_in_user_turn`, no transcript accumulation
4. Logs downstream frames only (same de-dup as today)

Transcripts (user turn text, bot turn text) are reconstructed in post-processing by joining `stt_final` tokens between `vad_started`/`user_turn_end` and `llm_token` tokens between `llm_started`/`llm_stopped`.

#### Turn probability

`PeriodicSmartTurnAnalyzer` logs remain as-is: `turn_probability` events emitted from the polling task. Bug C (sampling gap at pause) should be fixed — see Long-Pause Bugs section below.

---

### Tier 2 — High-Level State Reconstruction (post-processing only, implement after Tier 1 is validated)

**Goal:** Derive the coarse 6-state FSM from raw Tier 1 events. This is entirely offline; no changes to the observer are needed at this stage.

States and derivation rules (precedence order — highest wins):

| State | Onset | Exit |
|---|---|---|
| `user_speaking` | `vad_started` | `user_turn_end` |
| `user_interrupting` | `vad_started` while `bot_speaking_started` seen but not `bot_speaking_stopped` | `interruption` + `bot_speaking_stopped` |
| `processing` | `user_turn_end` | `bot_speaking_started` |
| `speaking` | `bot_speaking_started` | `bot_speaking_stopped` |
| `listening` | `bot_speaking_stopped` (or session start) | `vad_started` |
| `idle` | initial | `vad_started` or `bot_speaking_started` |

Sub-phases within `processing` (derivable from Tier 1 events):
- `llm_streaming`: `llm_started` → `llm_stopped`
- `tts_startup`: `tts_started` → `tts_first_audio` (per context)
- `audio_playback`: `bot_speaking_started` → `bot_speaking_stopped`

---

## Latency Reduction — Overlapping Pipeline Stages

### 1. Start LLM generation sooner (before / during user silence)

Currently: `VAD stops → SmartTurn model runs → user_turn_end → LLM starts`.

Three options, in order of implementation complexity:

| Option | Change | Latency saved | Risk |
|---|---|---|---|
| **A. Lower VAD `stop_secs`** | Reduce `SileroVADAnalyzer` `stop_secs` (default 0.8 s) to e.g. 0.3 s | ~0.5 s | More false turn-ends on natural pauses |
| **B. Early trigger from `PeriodicSmartTurnAnalyzer`** | In a custom turn stop strategy, fire `user_turn_end` when periodic inference returns `probability > threshold` (e.g. 0.85) *during active speech*, not only after VAD stops | ~200–400 ms | False positives mid-sentence; needs threshold tuning |
| **C. Speculative LLM on interim STT** | Feed `InterimTranscriptionFrame` text to LLM speculatively; discard if user continues; commit on `user_turn_end` | 300–800 ms | Significant complexity; wasted LLM calls on wrong intermediates |

**Recommended starting point:** Option A (one-line change), then profile with `plot_vad.py` to see if the SmartTurn model is already predicting correctly well before VAD stops (as seen in `session_20260511_101618_ddcc919e.jsonl` where `turn_probability` reached 0.97 at t=19.047 s, 235 ms before `user_turn_end` at t=19.282 s).


---

## TTFT Analysis — Observations from `session_20260511_101618_ddcc919e.jsonl`

### TTFT is the dominant latency bottleneck

TTFT (Time to First Token) is the gap between `llm_started` and the first `llm_token`. Measured values from the session:

| Turn | User said | `llm_started` | First token | **TTFT** | First audio |
|---|---|---|---|---|---|
| 1 (intro) | *(bot-initiated)* | 2.360 s | 6.391 s | **4031 ms** | 7.079 s |
| 2 | "Hi. I'm wondering when I should call you." | 19.282 s | 20.250 s | **968 ms** | 20.907 s |
| 3 | "Okay. Thanks." | 32.188 s | 34.079 s | **1891 ms** | 34.235 s |
| 4 | "That's it." | 42.125 s | 42.969 s | **844 ms** | 43.172 s |

**Why the gap exists:** Before the model can emit a single output token it must process the entire input context in one forward pass (prompt prefill). This is compute-bound on OpenAI's servers. Once prefill completes, tokens stream out rapidly (~15–50 ms/token) because decoding is memory-bandwidth bound.

**TTS is not the bottleneck:** After the first token arrives, TTS starts within 46–500 ms and produces first audio in ~110–330 ms. Switching to `TextAggregationMode.TOKEN` saves the sentence-buffering delay but TTFT dominates it entirely.

### TTFT is highly variable and not monotonically correlated with context length

Turn 3 ("Okay. Thanks.", shorter context) was 1891 ms — more than twice turn 4 ("That's it.", longer context) at 844 ms. Three plausible causes:

1. **API server non-determinism** — shared load, routing, and queuing on OpenAI's infrastructure produces 2x+ swings between consecutive requests in the same session. This is the most likely explanation.
2. **HTTP/2 connection state** — if the idle connection was torn down between turns and had to be re-established, that adds a full round-trip before prefill even starts.
3. **Prompt cache population** — OpenAI caches prompt prefixes in 128-token blocks (above 1024 tokens). Turn 3 may have been the first to populate a cache block that turn 4 then hit at a discount.

### Implications for latency reduction strategy

- **TTFT variance means you cannot rely on API speed** — structural improvements (Options A/B above) always help regardless of server conditions, making them higher-priority than hoping for fast API responses.
- **Overlapping prefill with user speech is high-value** — in turn 2, `vad_started` was at 17.438 s but `llm_started` was at 19.282 s. If the LLM had been triggered at `vad_started`, 1.844 s of prefill time could have overlapped with the user still speaking, and TTFT would have been invisible to the listener.
- **Speculative generation (Option C) addresses TTFT directly** — by starting the LLM before `user_turn_end`, the prefill phase is complete (or partially complete) by the time the final transcript is confirmed. The gating processor only needs to decide whether to *release* the buffered tokens, not wait for them.