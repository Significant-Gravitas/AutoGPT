import { useToast } from "@/components/molecules/Toast/use-toast";
import { useStableCallback } from "@/hooks/useStableCallback";
import { isKey } from "@/lib/keyboard";
import { isMicrophonePermissionDenied } from "@/lib/microphone";
import { isTouchDevice } from "@/lib/platform";
import React, { KeyboardEvent, useEffect, useRef, useState } from "react";
import { downloadRecording } from "../../voice/downloadRecording";
import { focusComposer } from "./helpers";

const MAX_RECORDING_DURATION = 2 * 60 * 1000; // 2 minutes in ms

interface Args {
  setValue: React.Dispatch<React.SetStateAction<string>>;
  disabled?: boolean;
  value: string;
  inputId?: string;
  isStreaming?: boolean;
}

export function useVoiceRecording({
  setValue,
  disabled = false,
  value,
  inputId,
  isStreaming = false,
}: Args) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Kept apart from `error`: a failed transcription is answered inline, next
  // to the audio it still holds, not by a toast that expires in five seconds.
  const [transcriptionError, setTranscriptionError] = useState<string | null>(
    null,
  );
  // Up to two minutes of speech. Dropping it on a transient 500 is the whole
  // bug — the user cannot get those two minutes back.
  const [failedRecording, setFailedRecording] = useState<Blob | null>(null);
  // Bumped when the user dismisses. An attempt they have already waved away
  // must not put the row back on screen when it finally fails.
  const attemptRef = useRef(0);
  const [elapsedTime, setElapsedTime] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const isRecordingRef = useRef(false);

  const [isSupported, setIsSupported] = useState(false);
  // Sending the draft as transcription context ships with the brain-dump
  // experience (Path B records a dump on top of Otto's intro text).

  const valueRef = useRef(value);

  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  useEffect(() => {
    setIsSupported(
      !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    );
  }, []);

  function clearTimer() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function cleanup() {
    clearTimer();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setAudioStream(null);
    }
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    setElapsedTime(0);
  }

  function handleTranscription(text: string) {
    setValue((prev) => {
      const trimmedPrev = prev.trim();
      if (trimmedPrev) {
        return `${trimmedPrev} ${text}`;
      }
      return text;
    });
  }

  async function transcribeAudio(audioBlob: Blob) {
    const attempt = attemptRef.current;
    setIsTranscribing(true);
    setError(null);
    // The previous failure stays on screen through a retry: clearing it here
    // would blink the row away and back, and take the Retry button with it.

    async function transcribe() {
      const formData = new FormData();
      formData.append("audio", audioBlob);
      const draft = valueRef.current.trim();
      if (draft) {
        formData.append("context", draft);
      }

      const response = await fetch("/api/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Transcription failed");
      }

      const data = await response.json();
      if (data.text) {
        handleTranscription(data.text);
      }
      setTranscriptionError(null);
      setFailedRecording(null);
    }

    await transcribe()
      .catch((err) => {
        console.error("Transcription error:", err);
        // Dismissed while this was in flight: the user is done with this
        // recording, so the failure has nobody to tell. The success path is
        // deliberately not gated the same way — if the words do arrive, the
        // user gets what they dictated rather than losing it twice.
        if (attempt !== attemptRef.current) return;
        const message =
          err instanceof Error && err.message
            ? err.message
            : "Transcription failed";
        setTranscriptionError(message);
        setFailedRecording(audioBlob);
      })
      .finally(() => {
        setIsTranscribing(false);
      });
  }

  /** Re-sends the recording that failed, byte for byte. */
  function retryTranscription() {
    if (!failedRecording || isTranscribing || isRecordingRef.current) return;
    void transcribeAudio(failedRecording);
  }

  function downloadFailedRecording() {
    if (failedRecording) downloadRecording(failedRecording);
  }

  function dismissTranscriptionError() {
    attemptRef.current += 1;
    setTranscriptionError(null);
    setFailedRecording(null);
  }

  function stopRecording() {
    if (mediaRecorderRef.current && isRecordingRef.current) {
      mediaRecorderRef.current.stop();
      isRecordingRef.current = false;
      setIsRecording(false);
      clearTimer();
    }
  }

  async function startRecording() {
    if (disabled || isRecordingRef.current || isTranscribing) return;

    setError(null);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setAudioStream(stream);

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: pickRecordingMimeType(),
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, {
          type: mediaRecorder.mimeType,
        });

        // Cleanup stream
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
          setAudioStream(null);
        }

        if (audioBlob.size > 0) {
          await transcribeAudio(audioBlob);
        }
      };

      mediaRecorder.start(1000); // Collect data every second

      // Only once a new recording is genuinely under way. Everything above
      // can still throw — a denied prompt, an unsupported mime type — and the
      // catch has no way to give the previous recording back.
      setTranscriptionError(null);
      setFailedRecording(null);

      isRecordingRef.current = true;
      setIsRecording(true);
      startTimeRef.current = Date.now();

      // Start elapsed time timer
      timerRef.current = setInterval(() => {
        const elapsed = Date.now() - startTimeRef.current;
        setElapsedTime(elapsed);

        // Auto-stop at max duration
        if (elapsed >= MAX_RECORDING_DURATION) {
          stopRecording();
        }
      }, 100);
    } catch (err) {
      console.error("Failed to start recording:", err);
      if (isMicrophonePermissionDenied(err)) {
        setError("Microphone permission denied");
      } else {
        setError("Failed to access microphone");
      }
      cleanup();
    }
  }

  function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  const { toast } = useToast();

  useEffect(() => {
    if (error) {
      toast({
        title: "Voice recording failed",
        description: error,
        variant: "destructive",
      });
    }
  }, [error, toast]);

  // Puts the caret back once the words land; focus on mount is useChatInput's.
  // Not on a phone: there focus opens the keyboard over the thread.
  const wasTranscribingRef = useRef(false);
  useEffect(() => {
    const hasFinished = wasTranscribingRef.current && !isTranscribing;
    wasTranscribingRef.current = isTranscribing;
    if (hasFinished && inputId && !isTouchDevice()) focusComposer(inputId);
  }, [isTranscribing, inputId]);

  function handleKeyDown(event: KeyboardEvent<HTMLElement>) {
    // Allow space to toggle recording (start when empty, stop when recording)
    if (isKey(event, " ") && !isTranscribing) {
      if (isRecordingRef.current) {
        // Stop recording on space
        event.preventDefault();
        stopRecording();
        return;
      } else if (!value.trim() && !isStreaming) {
        // Start recording on space when input is empty and not streaming
        // (mirrors the visual disabled state of the mic button during streaming)
        event.preventDefault();
        void startRecording();
        return;
      }
    }
    // Block all key events when recording (except space handled above)
    if (isRecordingRef.current) {
      event.preventDefault();
      return;
    }
    // Let PromptInputTextarea handle remaining keys (Enter → submit, etc.)
  }

  const showMicButton = isSupported;
  // Don't include isRecording in disabled state - we need key events to work
  // Text input is blocked via handleKeyDown instead.
  // isStreaming is intentionally excluded: users can type and queue messages
  // while a stream is in-flight; the stop button handles the streaming state.
  const isInputDisabled = disabled || isTranscribing;

  // Cleanup on unmount. Through a stable callback, so a new `cleanup` does not
  // re-run the effect and stop the recording.
  const cleanupOnUnmount = useStableCallback(cleanup);
  useEffect(() => {
    return () => {
      cleanupOnUnmount();
    };
  }, [cleanupOnUnmount]);

  return {
    isRecording,
    isTranscribing,
    error,
    transcriptionError,
    hasFailedRecording: failedRecording !== null,
    retryTranscription,
    downloadFailedRecording,
    dismissTranscriptionError,
    elapsedTime,
    startRecording,
    stopRecording,
    toggleRecording,
    isSupported,
    handleKeyDown,
    showMicButton,
    isInputDisabled,
    audioStream,
  };
}

function pickRecordingMimeType() {
  return MediaRecorder.isTypeSupported("audio/webm")
    ? "audio/webm"
    : "audio/mp4";
}
