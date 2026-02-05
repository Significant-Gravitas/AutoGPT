import { useToast } from "@/components/molecules/Toast/use-toast";
import React, {
  KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const MAX_RECORDING_DURATION = 2 * 60 * 1000; // 2 minutes in ms

interface Args {
  setValue: React.Dispatch<React.SetStateAction<string>>;
  disabled?: boolean;
  isStreaming?: boolean;
  value: string;
  baseHandleKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  inputId?: string;
}

export function useVoiceRecording({
  setValue,
  disabled = false,
  isStreaming = false,
  value,
  baseHandleKeyDown,
  inputId,
}: Args) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const isRecordingRef = useRef(false);

  const isSupported =
    typeof window !== "undefined" &&
    !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const cleanup = useCallback(() => {
    clearTimer();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    setElapsedTime(0);
  }, [clearTimer]);

  const handleTranscription = useCallback(
    (text: string) => {
      setValue((prev) => {
        const trimmedPrev = prev.trim();
        if (trimmedPrev) {
          return `${trimmedPrev} ${text}`;
        }
        return text;
      });
    },
    [setValue],
  );

  const transcribeAudio = useCallback(
    async (audioBlob: Blob) => {
      setIsTranscribing(true);
      setError(null);

      try {
        const formData = new FormData();
        formData.append("audio", audioBlob);

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
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Transcription failed";
        setError(message);
        console.error("Transcription error:", err);
      } finally {
        setIsTranscribing(false);
      }
    },
    [handleTranscription, inputId],
  );

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecordingRef.current) {
      mediaRecorderRef.current.stop();
      isRecordingRef.current = false;
      setIsRecording(false);
      clearTimer();
    }
  }, [clearTimer]);

  const startRecording = useCallback(async () => {
    if (disabled || isRecordingRef.current || isTranscribing) return;

    setError(null);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "audio/mp4",
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
        }

        if (audioBlob.size > 0) {
          await transcribeAudio(audioBlob);
        }
      };

      mediaRecorder.start(1000); // Collect data every second
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
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError("Microphone permission denied");
      } else {
        setError("Failed to access microphone");
      }
      cleanup();
    }
  }, [disabled, isTranscribing, stopRecording, transcribeAudio, cleanup]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

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

  useEffect(() => {
    if (!isTranscribing && inputId) {
      const inputElement = document.getElementById(inputId);
      if (inputElement) {
        inputElement.focus();
      }
    }
  }, [isTranscribing, inputId]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === " " && !value.trim() && !isTranscribing) {
        event.preventDefault();
        toggleRecording();
        return;
      }
      baseHandleKeyDown(event);
    },
    [value, isTranscribing, toggleRecording, baseHandleKeyDown],
  );

  const showMicButton = isSupported;
  const isInputDisabled = disabled || isStreaming || isTranscribing;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    isRecording,
    isTranscribing,
    error,
    elapsedTime,
    startRecording,
    stopRecording,
    toggleRecording,
    isSupported,
    handleKeyDown,
    showMicButton,
    isInputDisabled,
    audioStream: streamRef.current,
  };
}
