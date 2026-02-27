import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import {
  ArrowUpIcon,
  CircleNotchIcon,
  MicrophoneIcon,
  StopIcon,
} from "@phosphor-icons/react";
import { ChangeEvent, useCallback, useEffect, useState } from "react";
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
    handleKeyDown: baseHandleKeyDown,
    handleSubmit,
    handleChange: baseHandleChange,
    hasMultipleLines,
  } = useChatInput({
    onSend: async (message: string) => {
      await onSend(message, hasFiles ? files : undefined);
      // Only clear files after successful send (onSend throws on failure)
      setFiles([]);
    },
    disabled: isBusy,
    canSendEmpty: hasFiles,
    maxRows: 4,
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
    baseHandleKeyDown,
    inputId,
  });

  // Block text changes when recording
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      if (isRecording) return;
      baseHandleChange(e);
    },
    [isRecording, baseHandleChange],
  );

  function handleFilesSelected(newFiles: File[]) {
    setFiles((prev) => [...prev, ...newFiles]);
  }

  function handleRemoveFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  const isExpanded = hasMultipleLines || hasFiles;

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      <div className="relative">
        <div
          className={cn(
            "overflow-hidden border bg-white shadow-sm",
            "focus-within:ring-1",
            isRecording
              ? "border-red-400 focus-within:border-red-400 focus-within:ring-red-400"
              : "border-neutral-200 focus-within:border-zinc-400 focus-within:ring-zinc-400",
            isExpanded ? "rounded-xlarge" : "rounded-full",
          )}
        >
          <FileChips
            files={files}
            onRemove={handleRemoveFile}
            isUploading={isUploadingFiles}
          />
          <div id={`${inputId}-wrapper`} className="relative">
            {!value && !isRecording && !hasFiles && (
              <div
                className="pointer-events-none absolute inset-0 top-0.5 flex items-center justify-start pl-14 text-[1rem] text-zinc-400"
                aria-hidden="true"
              >
                {isTranscribing ? "Transcribing..." : placeholder}
              </div>
            )}
            <textarea
              id={inputId}
              aria-label="Chat message input"
              value={value}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              disabled={isInputDisabled}
              rows={1}
              className={cn(
                "w-full resize-none overflow-y-auto border-0 bg-transparent text-[1rem] leading-6 text-black",
                "placeholder:text-zinc-400",
                "focus:outline-none focus:ring-0",
                "disabled:text-zinc-500",
                showMicButton
                  ? "pb-4 pl-14 pr-[6.25rem] pt-4"
                  : "pb-4 pl-14 pr-14 pt-4",
              )}
            />
            {isRecording && !value && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <RecordingIndicator
                  elapsedTime={elapsedTime}
                  audioStream={audioStream}
                />
              </div>
            )}
          </div>
        </div>
        <span id="chat-input-hint" className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>

        <div className="absolute bottom-[7px] left-2 flex items-center gap-1">
          <AttachmentMenu
            onFilesSelected={handleFilesSelected}
            disabled={isBusy}
          />
        </div>

        <div className="absolute bottom-[7px] right-2 flex items-center gap-1">
          {showMicButton && (
            <Button
              type="button"
              variant="icon"
              size="icon"
              aria-label={isRecording ? "Stop recording" : "Start recording"}
              onClick={toggleRecording}
              disabled={isBusy || isTranscribing}
              className={cn(
                isRecording
                  ? "animate-pulse border-red-500 bg-red-500 text-white hover:border-red-600 hover:bg-red-600"
                  : isTranscribing
                    ? "border-zinc-300 bg-zinc-100 text-zinc-400"
                    : "border-zinc-300 bg-white text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50 hover:text-zinc-700",
                isStreaming && "opacity-40",
              )}
            >
              {isTranscribing ? (
                <CircleNotchIcon className="h-4 w-4 animate-spin" />
              ) : (
                <MicrophoneIcon className="h-4 w-4" weight="bold" />
              )}
            </Button>
          )}
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
                (isBusy || (!value.trim() && !hasFiles) || isRecording) &&
                  "opacity-20",
              )}
              disabled={isBusy || (!value.trim() && !hasFiles) || isRecording}
            >
              <ArrowUpIcon className="h-4 w-4" weight="bold" />
            </Button>
          )}
        </div>
      </div>
    </form>
  );
}
