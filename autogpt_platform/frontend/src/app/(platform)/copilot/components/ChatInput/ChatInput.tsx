import {
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { InputGroup } from "@/components/ui/input-group";
import { cn } from "@/lib/utils";
import { CircleNotchIcon, MicrophoneIcon } from "@phosphor-icons/react";
import { ChangeEvent } from "react";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { useChatInput } from "./useChatInput";
import { useVoiceRecording } from "./useVoiceRecording";

export interface Props {
  onSend: (message: string) => void | Promise<void>;
  disabled?: boolean;
  isStreaming?: boolean;
  onStop?: () => void;
  placeholder?: string;
  className?: string;
  inputId?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  isStreaming = false,
  onStop,
  placeholder = "Type your message...",
  className,
  inputId = "chat-input",
}: Props) {
  const {
    value,
    setValue,
    handleSubmit,
    handleChange: baseHandleChange,
  } = useChatInput({
    onSend,
    disabled: disabled || isStreaming,
    inputId,
  });

  const {
    isRecording,
    isTranscribing,
    elapsedTime,
    toggleRecording,
    handleKeyDown,
    showMicButton,
    isInputDisabled,
    audioStream,
  } = useVoiceRecording({
    setValue,
    disabled: disabled || isStreaming,
    isStreaming,
    value,
    inputId,
  });

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    if (isRecording) return;
    baseHandleChange(e);
  }

  const canSend = !disabled && !!value.trim() && !isRecording;

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      <InputGroup
        className={cn(
          "overflow-hidden has-[[data-slot=input-group-control]:focus-visible]:border-neutral-200 has-[[data-slot=input-group-control]:focus-visible]:ring-0",
          isRecording &&
            "border-red-400 ring-1 ring-red-400 has-[[data-slot=input-group-control]:focus-visible]:border-red-400 has-[[data-slot=input-group-control]:focus-visible]:ring-red-400",
        )}
      >
        <PromptInputBody className="relative block w-full">
          <PromptInputTextarea
            id={inputId}
            aria-label="Chat message input"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={isInputDisabled}
            placeholder={isTranscribing ? "Transcribing..." : placeholder}
          />
          {isRecording && !value && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <RecordingIndicator
                elapsedTime={elapsedTime}
                audioStream={audioStream}
              />
            </div>
          )}
        </PromptInputBody>

        <span id="chat-input-hint" className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>

        <PromptInputFooter>
          <PromptInputTools>
            {showMicButton && (
              <PromptInputButton
                aria-label={isRecording ? "Stop recording" : "Start recording"}
                onClick={toggleRecording}
                disabled={disabled || isTranscribing || isStreaming}
                className={cn(
                  "size-[2.625rem] rounded-[96px] border border-zinc-300 bg-transparent text-black hover:border-zinc-600 hover:bg-zinc-100",
                  isRecording &&
                    "animate-pulse border-red-500 bg-red-500 text-white hover:border-red-600 hover:bg-red-600",
                  isTranscribing && "bg-zinc-100 text-zinc-400",
                  isStreaming && "opacity-40",
                )}
              >
                {isTranscribing ? (
                  <CircleNotchIcon className="h-4 w-4 animate-spin" />
                ) : (
                  <MicrophoneIcon
                    className="h-4 w-4 text-zinc-500"
                    weight="bold"
                  />
                )}
              </PromptInputButton>
            )}
          </PromptInputTools>

          {isStreaming ? (
            <PromptInputSubmit status="streaming" onStop={onStop} />
          ) : (
            <PromptInputSubmit disabled={!canSend} />
          )}
        </PromptInputFooter>
      </InputGroup>
    </form>
  );
}
