# Logged Frame Reference

Brief descriptions of each frame/event captured by `ConversationLogObserver`, grouped by source service. Focus is on *why* the frame is triggered and what it implies about conversation state.

---

## VAD (Voice Activity Detection)

### `VADUserStartedSpeakingFrame` → `vad_started`
The VAD model detected the start of continuous audio energy from the user's microphone. The user has begun speaking (or a noise has started). The bot should treat itself as being listened to; any ongoing bot speech is a candidate for interruption.

### `VADUserStoppedSpeakingFrame` → `vad_stopped`
The VAD model detected that audio energy dropped below the speech threshold. The user has paused or stopped speaking. This does **not** yet mean the turn is over — the turn analyzer may decide the user is mid-sentence.

---

## Turn Detection

### `UserStoppedSpeakingFrame` → `user_turn_end`
The turn analyzer (e.g. `TurnAnalyzerUserTurnStopStrategy`) has decided the user's conversational turn is complete. This is the signal that triggers the LLM to generate a response. Everything between `vad_started` and `user_turn_end` constitutes one user turn.

### `MetricsFrame` (TurnMetricsData) → `turn_probability`
The smart-turn model has computed a probability that the user has finished speaking. Values near 1.0 mean the model is confident the turn is complete. Emitted both at VAD-stop (via `MetricsFrame` downstream) and periodically during speech (via `PeriodicSmartTurnAnalyzer`). Does not directly affect pipeline flow — it is observational.

---

## STT (Speech-to-Text)

### `InterimTranscriptionFrame` → `stt_interim`
A partial, unstable transcript of what the user said so far. Emitted frequently while the user is still speaking. Text may change or be retracted by a later interim or final result.

### `TranscriptionFrame` → `stt_final`
A stable, final transcript of a completed utterance segment. Emitted after the user pauses or stops. This is the text fed into the LLM context; joining all `stt_final` events between `vad_started` and `user_turn_end` reconstructs the full user turn.

---

## LLM (Language Model)

### `LLMFullResponseStartFrame` → `llm_started`
The LLM has begun streaming its response. The bot is now "thinking" / generating. No audio has been produced yet; the pipeline is in a processing/pre-speech state.

### `LLMFullResponseEndFrame` → `llm_stopped`
The LLM has finished generating its full response. Marks the end of the LLM streaming phase. TTS may still be running at this point.

### `LLMTextFrame` → `llm_token`
A single text token (or small chunk) from the LLM stream. High-frequency; concatenating all tokens between `llm_started` and `llm_stopped` reconstructs the full bot response text.

### `FunctionCallInProgressFrame` → `llm_function_call`
The LLM is invoking a tool/function instead of (or before) generating a spoken response. The bot is still in a processing state; the turn is extended until the function completes and the LLM continues.

---

## TTS (Text-to-Speech)

### `TTSStartedFrame` → `tts_started`
The TTS service has accepted a new synthesis request for a given `context_id`. Audio generation has begun but no audio data has arrived yet. Multiple contexts can be active simultaneously (e.g. sentence-by-sentence streaming).

### `TTSStoppedFrame` → `tts_stopped`
The TTS service has finished synthesising audio for a `context_id`. Paired with `tts_started`; the gap is the TTS synthesis latency for that chunk.

### `TTSAudioRawFrame` (first per context) → `tts_first_audio`
The first audio chunk for a given TTS `context_id` has arrived. This is the key latency marker: the time from `tts_started` to `tts_first_audio` is the TTS time-to-first-audio. Audio is now queued for playback.

---

## Transport / Playback

### `BotStartedSpeakingFrame` → `bot_speaking_started`
The transport layer has started sending audio to the user. The bot is now audibly speaking. This follows `tts_first_audio` once the audio buffer is drained to the output. The gap between `user_turn_end` and `bot_speaking_started` is the end-to-end response latency.

### `BotStoppedSpeakingFrame` → `bot_speaking_stopped`
The transport layer has finished sending all queued audio. The bot has finished its spoken turn. The pipeline returns to a listening state.

---

## Interruption

### `InterruptionFrame` → `interruption`
The user started speaking while the bot was still speaking (`vad_started` during bot speech), triggering an interruption. The bot stops speaking immediately; any queued LLM tokens and TTS audio are discarded. The pipeline returns to user-speaking state.

---

## Conversation State Summary

```
idle
  → vad_started          : user begins speaking
    → stt_interim*       : partial transcripts arrive
    → turn_probability*  : model assesses if turn is done
  → vad_stopped          : user pauses (may not be end of turn)
  → user_turn_end        : turn confirmed complete
    → llm_started        : LLM begins generating
      → llm_token*       : tokens stream in
      → tts_started      : TTS begins synthesis
        → tts_first_audio: first audio chunk ready
    → llm_stopped        : LLM done generating
  → bot_speaking_started : audio playback starts (audible response)
    → [interruption]     : user speaks over bot → back to vad_started
  → bot_speaking_stopped : bot done speaking → back to idle/listening
```

`*` = may repeat multiple times within a phase
