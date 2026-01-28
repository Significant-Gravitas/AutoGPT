import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import {
  ArrowUpIcon,
  CircleNotchIcon,
  MicrophoneIcon,
  StopIcon,
} from "@phosphor-icons/react";
import { KeyboardEvent, useCallback, useEffect } from "react";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { useChatInput } from "./useChatInput";
import { useVoiceRecording } from "./useVoiceRecording";

export interface Props {
  onSend: (message: string) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  onStop?: () => void;
  placeholder?: string;
  className?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  isStreaming = false,
  onStop,
  placeholder = "Type your message...",
  className,
}: Props) {
  const inputId = "chat-input";
  const {
    value,
    setValue,
    handleKeyDown: baseHandleKeyDown,
    handleSubmit,
    handleChange,
    hasMultipleLines,
  } = useChatInput({
    onSend,
    disabled: disabled || isStreaming,
    maxRows: 4,
    inputId,
  });

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

  const {
    isRecording,
    isTranscribing,
    error: voiceError,
    elapsedTime,
    toggleRecording,
    isSupported: isVoiceSupported,
  } = useVoiceRecording({
    onTranscription: handleTranscription,
    disabled: disabled || isStreaming,
  });

  const { toast } = useToast();

  // Show voice recording errors via toast
  useEffect(() => {
    if (voiceError) {
      toast({
        title: "Voice recording failed",
        description: voiceError,
        variant: "destructive",
      });
    }
  }, [voiceError, toast]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      // Space key toggles recording when input is empty
      if (event.key === " " && !value.trim() && !isTranscribing) {
        event.preventDefault();
        toggleRecording();
        return;
      }
      baseHandleKeyDown(event);
    },
    [value, isTranscribing, toggleRecording, baseHandleKeyDown],
  );

  const showMicButton = isVoiceSupported && !isStreaming;
  const isInputDisabled = disabled || isStreaming || isTranscribing;

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      <div className="relative">
        <div
          id={`${inputId}-wrapper`}
          className={cn(
            "relative overflow-hidden border bg-white shadow-sm",
            "focus-within:ring-1",
            isRecording
              ? "border-red-400 focus-within:border-red-400 focus-within:ring-red-400"
              : "border-neutral-200 focus-within:border-zinc-400 focus-within:ring-zinc-400",
            hasMultipleLines ? "rounded-xlarge" : "rounded-full",
          )}
        >
          <textarea
            id={inputId}
            aria-label="Chat message input"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={
              isTranscribing ? "Transcribing..." : isRecording ? "" : placeholder
            }
            disabled={isInputDisabled}
            rows={1}
            className={cn(
              "w-full resize-none overflow-y-auto border-0 bg-transparent text-[1rem] leading-6 text-black",
              "placeholder:text-zinc-400",
              "focus:outline-none focus:ring-0",
              "disabled:text-zinc-500",
              hasMultipleLines
                ? "pb-6 pl-4 pr-4 pt-2"
                : showMicButton
                  ? "pb-4 pl-14 pr-14 pt-4"
                  : "pb-4 pl-4 pr-14 pt-4",
            )}
          />
          {isRecording && !value && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <RecordingIndicator elapsedTime={elapsedTime} />
            </div>
          )}
        </div>
        <span id="chat-input-hint" className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>

        {showMicButton && (
          <div className="absolute bottom-[7px] left-2 flex items-center gap-1">
            <Button
              type="button"
              variant="icon"
              size="icon"
              aria-label={isRecording ? "Stop recording" : "Start recording"}
              onClick={toggleRecording}
              disabled={disabled || isTranscribing}
              className={cn(
                isRecording
                  ? "animate-pulse border-red-500 bg-red-500 text-white hover:border-red-600 hover:bg-red-600"
                  : isTranscribing
                    ? "border-zinc-300 bg-zinc-100 text-zinc-400"
                    : "border-zinc-300 bg-white text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50 hover:text-zinc-700",
              )}
            >
              {isTranscribing ? (
                <CircleNotchIcon className="h-4 w-4 animate-spin" />
              ) : (
                <MicrophoneIcon className="h-4 w-4" weight="bold" />
              )}
            </Button>
          </div>
        )}

        <div className="absolute bottom-[7px] right-2 flex items-center gap-1">
          {isStreaming ? (
            <Button
              type="button"
              variant="icon"
              size="icon"
              aria-label="Stop generating"
              onClick={onStop}
              className="border-red-600 bg-red-600 text-white hover:border-red-800 hover:bg-red-800"
            >
              <StopIcon className="h-4 w-4" weight="bold" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="icon"
              size="icon"
              aria-label="Send message"
              className={cn(
                "border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900",
                (disabled || !value.trim() || isRecording) && "opacity-20",
              )}
              disabled={disabled || !value.trim() || isRecording}
            >
              <ArrowUpIcon className="h-4 w-4" weight="bold" />
            </Button>
          )}
        </div>
      </div>
    </form>
  );
}
