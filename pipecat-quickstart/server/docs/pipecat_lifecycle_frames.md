# Pipecat Lifecycle Frames Research

## Key Findings

### Frame Categories and Lifecycle Patterns

**STT (Speech-to-Text):**
- Start frame pattern: `VADUserStartedSpeakingFrame` → triggers STT processing
- Active frames: `InterimTranscriptionFrame` (partial results), `TranscriptionFrame` (final)
- End frame pattern: `VADUserStoppedSpeakingFrame` → signals end of input
- Finalize: `TranscriptionFrame.finalized=True` marks the final result
- Single exclusive active state: YES (one STT stream active at a time)
- Clear start/end pairing: YES (VAD provides signals)

**LLM (Language Model):**
- Start frame: `LLMFullResponseStartFrame` (pushed by LLMService.process_frame when LLMContextFrame arrives)
- Active frames: `LLMTextFrame` (token stream), `LLMThoughtStartFrame`/`LLMThoughtTextFrame`/`LLMThoughtEndFrame` (optional)
- End frame: `LLMFullResponseEndFrame` (always pushed in finally block)
- Optional triggers: `LLMRunFrame` (trigger context processing), `LLMContextSummaryRequestFrame` (for context compression)
- Single exclusive active state: YES (one LLM response at a time)
- Clear start/end pairing: YES (strict finally-block guarantee)
- Note: `FunctionCallInProgressFrame` (uninterruptible) signals ongoing function execution

**TTS (Text-to-Speech):**
- Start frame: `TTSStartedFrame` (context_id embedded)
- Active frames: `TTSTextFrame` (aggregated sentences/tokens), `TTSAudioRawFrame` (audio chunks)
- End frame: `TTSStoppedFrame` (context_id embedded, matches TTSStartedFrame)
- Context lifecycle: Each context_id gets one TTSStartedFrame → multiple audio frames → one TTSStoppedFrame
- Single exclusive active state: YES per context (but multiple contexts can co-exist via context_id)
- Clear start/end pairing: YES (context_id ties start/stop together)
- Pattern: `LLMFullResponseStartFrame` → creates turn_context_id → one or more TTSStarted/Audio/Stopped sequences

**VAD/Turn Management:**
- VAD frames: `VADUserStartedSpeakingFrame` (with start_secs, timestamp), `VADUserStoppedSpeakingFrame` (with stop_secs, timestamp)
- User turn frames: `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`
- Transport speaking frames: `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `BotSpeakingFrame`
- Clear pairing: YES for each frame type

**Interruption/Transport:**
- Interruption signal: `InterruptionFrame` (system frame, carries optional asyncio.Event)
- Transport frames: Bot speaking lifecycle tracked via BotStartedSpeakingFrame/BotStoppedSpeakingFrame
- No clear matching for InterruptionFrame (it's a signal, not a paired lifecycle)

---

## Concurrency Analysis

### Can states overlap?
- STT + LLM: YES (STT can be running while LLM completes)
- STT + TTS: YES (can happen during interruptions or simultaneous I/O)
- LLM + TTS: YES (TTS begins consuming as LLM streams tokens)
- Multiple TTS contexts: YES (by design, via context_id mechanism)
- LLM function calls + text: YES (FunctionCallInProgressFrame is uninterruptible)

### Strict exclusivity:
- Per-service: NO (concurrent active states expected across services)
- Per-context (TTS): YES (one context_id = one exclusive sequence)
- Per-LLM response: YES (LLMFullResponseStartFrame → LLMFullResponseEndFrame is exclusive)
- Per-STT utterance: YES (VADUserStartedSpeakingFrame → VADUserStoppedSpeakingFrame is exclusive)

---

## Frames Without Clear Matching "End" Signal

1. **`InterruptionFrame`** — signals an interruption but has no matching "InterruptionEnded" frame
   - Completion tracked via optional `frame.event: asyncio.Event` if caller awaits it
   - Processor must call `frame.complete()` if it stops propagation to avoid stalls

2. **`LLMRunFrame`** — triggers LLM processing but doesn't pair with anything
   - Response is bracketed by `LLMFullResponseStartFrame`/`LLMFullResponseEndFrame` instead

3. **`FunctionCallInProgressFrame`** — marks execution start but no explicit "FunctionCallCompleted" frame
   - Completion tracked via context updates when result arrives (via `FunctionCallResultFrame`)

4. **`LLMContextSummaryRequestFrame`** — sent to LLM for async processing
   - Response comes asynchronously via `LLMContextSummaryResultFrame`

---

## Key Architectural Patterns

1. **Uninterruptible Frames**: `EndFrame`, `StopFrame`, `FunctionCallInProgressFrame`, `LLMContextSummaryResultFrame`, `FunctionCallResultFrame`
   - These survive interruptions and are never dropped

2. **Context IDs**: TTS uses context_id to multiplex concurrent outputs
   - Each context_id gets its own TTSStartedFrame/TTSStoppedFrame pair
   - Enables parallel text-to-speech processing (sentence-level pipelining)

3. **Async Responses**: Some frames get async responses (summarization, function calls)
   - Request frame paired with result frame asynchronously

4. **Event Signaling**: InterruptionFrame optionally carries asyncio.Event for completion tracking

---

## Full Emission Sequences

### STT (Deepgram)
```
VADUserStartedSpeakingFrame
  → InterimTranscriptionFrame  (0..n)
  → TranscriptionFrame(finalized=True)
  → VADUserStoppedSpeakingFrame
```

### LLM (OpenAI)
```
LLMFullResponseStartFrame
  → LLMTextFrame  (0..n tokens)
  → [optional: FunctionCallInProgressFrame → FunctionCallResultFrame]
  → LLMFullResponseEndFrame   ← always emitted in finally block
```

### TTS (Cartesia) — per context_id
```
TTSStartedFrame(context_id=X)
  → TTSTextFrame(context_id=X)  (1..n sentences)
  → TTSAudioRawFrame(context_id=X)  (1..n chunks)
  → TTSStoppedFrame(context_id=X)
```
Multiple context_ids can interleave in the frame stream.

### Transport output
```
BotStartedSpeakingFrame
  → [audio playback]
  → BotStoppedSpeakingFrame
```
