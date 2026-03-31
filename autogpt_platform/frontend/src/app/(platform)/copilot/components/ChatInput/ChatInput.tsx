import {
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { InputGroup } from "@/components/ui/input-group";
import { cn } from "@/lib/utils";
import { ChangeEvent, useEffect, useState } from "react";
import { AttachmentMenu } from "./components/AttachmentMenu";
import { FileChips } from "./components/FileChips";
import { RecordingButton } from "./components/RecordingButton";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { useCopilotUIStore } from "../../store";
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
  const { copilotMode, setCopilotMode } = useCopilotUIStore();
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

  const resolvedPlaceholder = isRecording
    ? ""
    : isTranscribing
      ? "Transcribing..."
      : placeholder;

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
            placeholder={resolvedPlaceholder}
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
            <button
              type="button"
              onClick={() =>
                setCopilotMode(
                  copilotMode === "extended_thinking"
                    ? "fast"
                    : "extended_thinking",
                )
              }
              className={cn(
                "inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors",
                copilotMode === "extended_thinking"
                  ? "bg-purple-100 text-purple-700 hover:bg-purple-200 dark:bg-purple-900/30 dark:text-purple-300"
                  : "bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900/30 dark:text-amber-300",
              )}
              title={
                copilotMode === "extended_thinking"
                  ? "Extended Thinking mode — deeper reasoning (click to switch to Fast mode)"
                  : "Fast mode — quicker responses (click to switch to Extended Thinking)"
              }
            >
              {copilotMode === "extended_thinking" ? (
                <>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
                    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
                    <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
                    <path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
                    <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
                    <path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
                    <path d="M19.938 10.5a4 4 0 0 1 .585.396" />
                    <path d="M6 18a4 4 0 0 1-1.967-.516" />
                    <path d="M19.967 17.484A4 4 0 0 1 18 18" />
                  </svg>
                  Thinking
                </>
              ) : (
                <>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z" />
                  </svg>
                  Fast
                </>
              )}
            </button>
          </PromptInputTools>

          <div className="flex items-center gap-4">
            {showMicButton && (
              <RecordingButton
                isRecording={isRecording}
                isTranscribing={isTranscribing}
                isStreaming={isStreaming}
                disabled={disabled || isTranscribing || isStreaming}
                onClick={toggleRecording}
              />
            )}
            {isStreaming ? (
              <PromptInputSubmit status="streaming" onStop={onStop} />
            ) : (
              <PromptInputSubmit disabled={!canSend} />
            )}
          </div>
        </PromptInputFooter>
      </InputGroup>
    </form>
  );
}
