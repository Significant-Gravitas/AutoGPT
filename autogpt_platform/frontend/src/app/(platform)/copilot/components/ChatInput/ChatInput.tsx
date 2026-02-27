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
import { ChangeEvent, useEffect, useState } from "react";
import { AttachmentMenu } from "./components/AttachmentMenu";
import { FileChips } from "./components/FileChips";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { useChatInput } from "./useChatInput";
import { useVoiceRecording } from "./useVoiceRecording";

export interface Props {
  onSend: (message: string, files?: File[]) => void | Promise<void>;
  disabled?: boolean;
  isStreaming?: boolean;
  isUploadingFiles?: boolean;
  onStop?: () => void;
  placeholder?: string;
  className?: string;
  inputId?: string;
  /** Files dropped onto the chat window by the parent. */
  droppedFiles?: File[];
  /** Called after droppedFiles have been merged into internal state. */
  onDroppedFilesConsumed?: () => void;
}

export function ChatInput({
  onSend,
  disabled = false,
  isStreaming = false,
  isUploadingFiles = false,
  onStop,
  placeholder = "Type your message...",
  className,
  inputId = "chat-input",
  droppedFiles,
  onDroppedFilesConsumed,
}: Props) {
  const [files, setFiles] = useState<File[]>([]);

  // Merge files dropped onto the chat window into internal state.
  useEffect(() => {
    if (droppedFiles && droppedFiles.length > 0) {
      setFiles((prev) => [...prev, ...droppedFiles]);
      onDroppedFilesConsumed?.();
    }
  }, [droppedFiles, onDroppedFilesConsumed]);

  const hasFiles = files.length > 0;
  const isBusy = disabled || isStreaming || isUploadingFiles;

  const {
    value,
    setValue,
    handleSubmit,
    handleChange: baseHandleChange,
  } = useChatInput({
    onSend: async (message: string) => {
      await onSend(message, hasFiles ? files : undefined);
      // Only clear files after successful send (onSend throws on failure)
      setFiles([]);
    },
    disabled: isBusy,
    canSendEmpty: hasFiles,
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
    disabled: isBusy,
    isStreaming,
    value,
    inputId,
  });

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    if (isRecording) return;
    baseHandleChange(e);
  }

  const canSend =
    !disabled &&
    (!!value.trim() || hasFiles) &&
    !isRecording &&
    !isTranscribing;

  function handleFilesSelected(newFiles: File[]) {
    setFiles((prev) => [...prev, ...newFiles]);
  }

  function handleRemoveFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      <InputGroup
        className={cn(
          "overflow-hidden has-[[data-slot=input-group-control]:focus-visible]:border-neutral-200 has-[[data-slot=input-group-control]:focus-visible]:ring-0",
          isRecording &&
            "border-red-400 ring-1 ring-red-400 has-[[data-slot=input-group-control]:focus-visible]:border-red-400 has-[[data-slot=input-group-control]:focus-visible]:ring-red-400",
        )}
      >
        <FileChips
          files={files}
          onRemove={handleRemoveFile}
          isUploading={isUploadingFiles}
        />
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

        <span id={`${inputId}-hint`} className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>

        <PromptInputFooter>
          <PromptInputTools>
            <AttachmentMenu
              onFilesSelected={handleFilesSelected}
              disabled={isBusy}
            />
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
                  <MicrophoneIcon className="h-4 w-4" weight="bold" />
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
