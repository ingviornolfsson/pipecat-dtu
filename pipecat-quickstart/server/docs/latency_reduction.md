# Reducing Bot Response Latency

## Goal

Reduce the perceived response latency of the voice bot. Measurements from session logs confirm that
**LLM processing is the bottleneck**. The current sequential pipeline is:

```
VAD_stop → STT finalizes (+100–200 ms) → LLM starts → TTFT (+300–500 ms) → TTS → audio
```

Typical total delay from VAD stop to first spoken word: **600–900 ms**. The STT finalization and
LLM time-to-first-token (TTFT) are fully sequential today — they can be overlapped.

---

## Overview of Proposals

### 1. Speculative Pre-generation with Adaptive Commit

Start the LLM call early (before the STT final transcript arrives), buffer its output, and commit or
abort the buffer depending on how much the final transcript differs from the interim used as input.

| | |
|---|---|
| **Expected gain** | ~300–400 ms average (saves most of LLM TTFT on successful speculations) |
| **Estimated hit rate** | 70–80 % for typical short conversational exchanges |
| **Net average gain** | ~250–300 ms per turn |
| **Risk** | Committing a response generated from a semantically wrong interim |
| **Main cost** | Wasted LLM tokens on aborted calls (~20–30 % of turns) |

The key design choices within this approach are (a) **when to trigger** the speculative call and (b)
**how to decide** whether to commit or abort. These are covered in detail below.

---

### 2. Decision Function Variants (sub-proposals)

All variants sit at the same point: when the STT final transcript arrives, decide whether the
speculative buffer should be flushed to TTS or discarded.

| Variant | Mechanism | Pros | Cons |
|---|---|---|---|
| **Word overlap** | Jaccard / recall between interim and final word sets | Simple, zero latency | Fails on right-branching sentences; high false-positive rate on the most dangerous cases |
| **Prefix + content-word check** | Final must extend interim as a prefix; new words must all be function words | Fast (regex-level), no model needed | Requires curated function-word list; fails when final rewrites rather than extends |
| **Surprisal via local LM** | -log P(new\_tokens \| interim\_context) with a small local model (Pythia-70M / GPT-2-small) | Semantically grounded; works even for rewritten finals | Adds 10–20 ms, requires bundled model (~300 MB) |
| **Surprisal via speculative LLM KV cache** | Extend the existing speculative call's KV cache with the final transcript; compare P(first\_response\_token) between interim and final inputs | Uses full conversation context; no extra model; near-zero added latency | Requires API support for KV cache reuse across requests (partial with OpenAI prompt caching) |

**Recommendation**: Start with the prefix + content-word check (implementable in one sitting) and
instrument it to log false positives. Use the logged cases to calibrate a threshold for the
surprisal-based check once the basic pipeline is working.

---

## Section 1 — Speculative Pre-generation: Mechanism and Pipeline Integration

### How it works

```
VAD_stop (t=0)
  │
  ├─► [snapshot last_interim from Deepgram]
  │    └─► start speculative LLM call with [system + history + last_interim]
  │         LLM tokens stream into SpeculativeBufferProcessor ──────────────►
  │
  └─► STT finalizes (+100–200 ms)
        TranscriptionFrame (final) arrives
        └─► compare final vs last_interim
              MATCH → flush buffer → TTS → audio  (near-zero added latency)
              NO MATCH → discard buffer
                         start real LLM call (KV cache covers history prefix)
```

On a successful match the latency saving is approximately the LLM TTFT, since that work was done
during the STT finalization window.

### Current pipeline

```python
Pipeline([
    transport.input(),
    stt,
    user_aggregator,   # waits for UserStoppedSpeakingFrame, then sends context to LLM
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
```

`VADOnlyUserTurnStopStrategy` fires `trigger_user_turn_stopped()` on the first `TranscriptionFrame`
after each VAD segment, so the LLM only starts after the final transcript is available.

### Required pipeline changes

```python
Pipeline([
    transport.input(),
    stt,
    speculative_orchestrator,   # NEW — intercepts VADUserStoppedSpeakingFrame + TranscriptionFrame
    user_aggregator,
    llm,
    speculative_gate,           # NEW — suppresses LLMRunFrame if speculation already confirmed
    speculative_buffer,         # NEW — holds LLM output until commit/abort decision
    tts,
    transport.output(),
    assistant_aggregator,
])
```

### SpeculativeOrchestrator (outside the pipeline or as a processor)

Responsibilities:
1. Keep a rolling `_last_interim: str` updated on every `InterimTranscriptionFrame`.
2. On `VADUserStoppedSpeakingFrame`: snapshot `last_interim`, launch a background task that calls
   the OpenAI API directly with `[system + history + last_interim]`, streams tokens into
   `SpeculativeBufferProcessor`.
3. On `TranscriptionFrame` (final): run the decision function against `last_interim`. Set buffer
   state to COMMIT or ABORT.

### SpeculativeBufferProcessor (between llm and tts)

Two states: `BUFFERING` and `PASSTHROUGH`.

```python
class SpeculativeBufferProcessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        if self._state == BUFFERING:
            self._buffer.append((frame, direction))
        else:
            await self.push_frame(frame, direction)

    async def commit(self):
        for frame, direction in self._buffer:
            await self.push_frame(frame, direction)
        self._buffer.clear()
        self._state = PASSTHROUGH

    async def abort(self):
        self._buffer.clear()
        self._state = PASSTHROUGH   # let the real LLM call flow through normally
```

### SpeculativeGate (between user_aggregator and llm)

When speculation is committed, the gate swallows the subsequent `LLMRunFrame` so the normal LLM
call does not also fire and produce a duplicate response.

### Coordination challenge

The speculative call runs *outside* the pipeline (direct API call in a background task), while the
normal `llm` processor sits *inside* the pipeline. Synchronisation is via shared state on
`SpeculativeBufferProcessor`: if buffer is committed before `LLMRunFrame` reaches the gate, the
gate drops the frame. If not committed in time, the gate passes it through and the buffer is in
ABORT state, discarding any late-arriving speculative tokens.

---

## Section 2 — Trigger Point: When to Start Speculating

The trigger point determines the quality and timing of the interim used for the speculative call.

### Background: Deepgram already uses forced finalization

Pipecat's Deepgram STT service (`src/pipecat/services/deepgram/stt.py:735`) sends
`{"type": "Finalize"}` to Deepgram on every `VADUserStoppedSpeakingFrame`. Deepgram responds with
a final transcript tagged `from_finalize=True`, which pipecat surfaces as
`TranscriptionFrame(finalized=True)`. The 50–150 ms delay observed in session logs is this
forced-finalization round-trip — it is already the fastest Deepgram can respond. There is nothing
left to speed up on the STT side.

This means **"Option B" (wait for a post-VAD-stop interim) is the same as waiting for the finalized
transcript**, which is what the current pipeline already does. It saves nothing. Only starting the
speculative call *at* VAD stop provides a genuine time saving.

### Commit/abort trigger

The decision function must fire on `TranscriptionFrame(finalized=True)`, not on the first
`TranscriptionFrame` after VAD stop. Non-finalized `TranscriptionFrame`s that arrive before
finalization are ordinary ongoing interims and should be ignored for the commit/abort decision:

```python
if isinstance(frame, TranscriptionFrame) and frame.finalized:
    decision = is_valid(last_interim, frame.text)
    await buffer.commit() if decision else await buffer.abort()
```

### Option A — Trigger at VAD stop (`VADUserStoppedSpeakingFrame`)

Start the speculative LLM call at VAD stop using the last pre-stop interim. The speculative window
is ~50–150 ms (the forced-finalization round-trip), during which the LLM begins generating tokens.

**Problem observed in logs**: The last Deepgram interim before VAD stop often lags the actual audio
by **500–1000 ms**. Examples from session logs:

- VAD stop at t=16328 ms, last interim at t=15390 ms → interim is "...shoes for" (missing "an outfit")
- VAD stop at t=14671 ms, last interim at t=13968 ms → interim is "making it difficult" (missing "decision")

The trailing gap means the interim at VAD stop can be missing the most semantically important words,
increasing the abort rate. The decision function (Section 3) must be strict enough to catch these.

Expected saving: ~50–150 ms.

### Option A+ — Rolling speculation on each interim [recommended]

Rather than waiting until VAD stop, start a speculative call on the first interim during speech.
When a new interim arrives that is meaningfully longer, abort the current speculative call and
restart with the updated transcript. By the time VAD fires and finalization completes, the speculative
call has been running for much longer — potentially the full duration of the last 200–400 ms interim
window rather than just the 50–150 ms finalization delay.

```
t=0:    InterimTranscriptionFrame "I want some help"        → start speculative call A
t=250:  InterimTranscriptionFrame "I want some help making" → restart as call B (abort A)
t=500:  InterimTranscriptionFrame "I want some help making a difficult" → restart as call C
t=650:  VADUserStoppedSpeakingFrame                         → nothing extra needed
t=750:  TranscriptionFrame(finalized=True) "...decision."  → commit/abort C
```

Call C has been running for ~250 ms by finalization, vs ~100 ms with Option A.

**Restart decision**: restart if the new interim is ≥ N words longer than the current speculation
basis (N=2 is a reasonable starting point — avoids restarting on punctuation-only updates). The
restart cost is low: aborted calls discard < 1 token of output since interims arrive faster than
LLM TTFT, and OpenAI prompt caching means re-processing history is negligible.

**Deepgram interim frequency**: Deepgram does not expose a parameter to control interim emission
frequency — it is driven by its internal acoustic processing. Observed cadence in session logs is
~200–400 ms. There is no way to increase this from the API side, so rolling speculation works with
whatever cadence Deepgram naturally provides.

Expected saving: ~300–500 ms (most of the TTFT hidden behind speech time), subject to abort rate.

### Option B — Trigger on SmartTurn high-probability event

The `PeriodicSmartTurnAnalyzer` (already in the codebase) computes turn-completion probability
every 200 ms during speech. In session_e0dce165, the model gave probability 0.625 at t=28781 ms,
which is before VAD stopped at t=29125 ms.

Using a SmartTurn probability threshold (e.g., ≥ 0.6) as the trigger would start speculation
*before* VAD fires, giving more time for the speculative call to complete. However, the transcript
at that point is still a partial interim, so the decision function challenge is more severe.

This option is best explored after the basic pipeline is working with Option A.

---

## Section 3 — Decision Function: How to Decide Commit vs Abort

### 3a — Word Overlap (rejected)

```python
def is_valid(interim, final, threshold=0.75):
    interim_words = set(interim.lower().split())
    final_words = set(final.lower().split())
    overlap = len(interim_words & final_words) / max(len(interim_words), 1)
    length_ratio = len(final_words) / max(len(interim_words), 1)
    return overlap >= threshold and length_ratio <= 1.5
```

**Why this fails**: English speech is right-branching — the content words that define the sentence's
meaning tend to arrive at the end. Word overlap is high precisely in the most dangerous cases
because all interim words appear in the final, but the *new* words change the meaning:

- Interim: `"I want some help making it difficult"` → word overlap with final = **0.86**
- Final: `"I want some help making a difficult decision."` — "decision" completely changes the task
- The 0.86 score would COMMIT, producing a wrong response

The length ratio check helps only for trivially short interims ("Boots." → "Boots, I think.") and
does nothing for the semantically dangerous cases.

**Do not use as the sole decision function.**

### 3b — Prefix + Content-Word Expansion Check (recommended starting point)

Core idea: the final must extend the interim as a *prefix*, and the new words must all be function
words (prepositions, auxiliaries, hedges). If any new word is a content word, abort.

```python
FUNCTION_WORDS = {
    'a', 'an', 'the', 'i', 'me', 'my', 'you', 'it', 'its', 'with', 'to',
    'of', 'in', 'on', 'at', 'by', 'for', 'and', 'but', 'or', 'so', 'yet',
    'just', 'well', 'then', 'now', 'think', 'guess', 'know', 'mean',
    'that', 'this', 'there', 'here', 'is', 'are', 'was', 'were', 'will',
    'do', 'did', 'have', 'had', 'be', 'been', 'being', 'not', 'no',
}

def is_valid(interim: str, final: str) -> bool:
    interim_words = interim.lower().split()
    final_words = final.lower().split()

    # Final must start with the same words as interim (prefix extension)
    if final_words[:len(interim_words)] != interim_words:
        return False

    # New words beyond the interim must all be function words
    added = [w.strip('.,!?;:') for w in final_words[len(interim_words):]]
    return all(w in FUNCTION_WORDS or not w for w in added)
```

Applied to log examples:
- `"making it difficult"` → `"making a difficult decision."` : words diverge at index 3
  ("it" vs "a") → **reject** ✓
- `"help me"` → `"help me with."` : "with" is in FUNCTION\_WORDS → **commit** ✓
- `"shoes for"` → `"shoes for an outfit."` : "outfit" is not a function word → **reject** ✓
- `"Boots."` → `"Boots, I think."` : "i", "think" both in FUNCTION\_WORDS → **commit** (minor
  hedge, acceptable) ✓

**Limitation**: fails when the final *rewrites* the interim rather than extending it (can happen
with ASR corrections). Should log all ABORT decisions for inspection.

### 3c — Surprisal-Based Decision Function

**Background**: Token surprisal = -log P(token | preceding context). Low surprisal means the token
was predictable given what came before; high surprisal means it was unexpected.

Applied to our problem: after the final arrives, compute the surprisal of the *new tokens* (those
in the final but not in the interim) given the interim as context. If surprisal is low, the LLM
would generate essentially the same response. If high, the new tokens materially change the input.

This directly addresses the word-overlap failure cases:
- P(`"decision"` | `"...making it difficult"`) is LOW — "decision" is an unexpected continuation
  after "difficult" already reads as a sentence-final adjective → high surprisal → **reject** ✓
- P(`"with"` | `"...help me"`) is HIGH — "with" is the most natural continuation → low surprisal
  → **commit** ✓

The same computation (token entropy at a potential utterance boundary) is used in the Byungdoh
et al. work (see `docs/todo.md`) for turn-end detection. There it measures "is there a local
entropy minimum here, suggesting a turn boundary?" Here it measures "were the additional tokens
predictable?" Both are instances of the same underlying surprisal signal.

#### Implementation option A — Local small LM (10–20 ms latency)

Run Pythia-70M or GPT-2-small locally to compute -log P(new\_tokens | interim\_context). These
models are ~300 MB and can run in ~10–20 ms on CPU for short sequences.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SurprisalDecider:
    def __init__(self, model_name="EleutherAI/pythia-70m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def is_valid(self, interim: str, final: str, threshold: float = 2.0) -> bool:
        if not final.startswith(interim.rstrip('.!?')):
            return False
        new_text = final[len(interim):].strip()
        if not new_text:
            return True
        input_ids = self.tokenizer(interim, return_tensors="pt").input_ids
        new_ids = self.tokenizer(new_text, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1]
        surprisal = -torch.log_softmax(logits, dim=-1)[new_ids[0]].item()
        return surprisal < threshold
```

This only checks the first new token's surprisal — sufficient for most cases since the highest-
information word in the continuation tends to be the first unexpected one.

**Caveat**: The local model has no conversation context (only the user utterance). This means it
misses domain-specific priors from the system prompt and prior turns.

#### Implementation option B — Surprisal via speculative LLM's KV cache (near-zero cost)

When the speculative call is running, OpenAI has already processed `[system + history + interim]`
into its KV cache. When the final arrives:

1. Make a new API call with `[system + history + final]` and `max_tokens=1` + `logprobs=True`.
2. The history prefix is served from cache (fast and cheap).
3. Compare the top-1 token probability distribution from the interim call vs the final call.
4. If KL divergence between the two first-token distributions is below a threshold → commit.

This uses the full conversation context and avoids a separate local model. Cost is one short API
call (~50 ms, mostly network) for which the history is already cached.

**Practical note**: OpenAI's prompt caching covers prefixes of ≥1024 tokens in 128-token blocks.
For short conversations this may not be cached yet. For longer conversations (3+ turns) it reliably
is.

---

## Summary and Recommended Implementation Order

1. **Implement SpeculativeBufferProcessor** as a passthrough by default. Wire it into the pipeline
   between `llm` and `tts`. Verify no regressions.

2. **Add SpeculativeOrchestrator** to track `last_interim` and trigger background LLM calls on each
   `InterimTranscriptionFrame` (rolling speculation). Restart when new interim is ≥ 2 words longer
   than the current basis. Commit/abort fires on `TranscriptionFrame(finalized=True)`.

3. **Add SpeculativeGate** between `user_aggregator` and `llm` to suppress duplicate calls when
   buffer is committed.

4. **Decision function**: start with the prefix + content-word check (Section 3b). Log every
   COMMIT and ABORT decision with the interim/final pair to the existing JSONL logger.

5. **Evaluate** hit rate and false-positive rate from the logs after a few sessions. If false-
   positive rate is acceptable (< 5%), ship. If not, layer in the surprisal check (Section 3c,
   Option A) on top.

6. **Long-term**: once the surprisal-based local LM is in place for the commit decision, consider
   reusing it for the Gate 1 trigger decision (entropy at VAD stop to decide whether to speculate
   at all) and for the SmartTurn replacement noted in `docs/todo.md`.
