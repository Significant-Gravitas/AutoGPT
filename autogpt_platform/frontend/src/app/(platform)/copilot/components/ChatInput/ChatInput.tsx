import {
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { toast } from "@/components/molecules/Toast/use-toast";
import { InputGroup } from "@/components/ui/input-group";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { Tray } from "@phosphor-icons/react";
import { ChangeEvent, useEffect, useState } from "react";
import { AttachmentMenu } from "./components/AttachmentMenu";
import { DryRunToggleButton } from "./components/DryRunToggleButton";
import { FileChips } from "./components/FileChips";
import { ModeToggleButton } from "./components/ModeToggleButton";
import { RecordingButton } from "./components/RecordingButton";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { useCopilotUIStore } from "../../store";
import { useChatInput } from "./useChatInput";
import { useVoiceRecording } from "./useVoiceRecording";

interface Props {
  onSend: (message: string, files?: File[]) => void | Promise<void>;
  disabled?: boolean;
  isStreaming?: boolean;
  isUploadingFiles?: boolean;
  onStop?: () => void;
  /** Called to enqueue a message when copilot is streaming and user has typed text. */
  onEnqueue?: (message: string) => void | Promise<void>;
  placeholder?: string;
  className?: string;
  inputId?: string;
  /** Files dropped onto the chat window by the parent. */
  droppedFiles?: File[];
  /** Called after droppedFiles have been merged into internal state. */
  onDroppedFilesConsumed?: () => void;
  /** When true, the dry-run toggle is disabled (session is active and immutable). */
  hasSession?: boolean;
}

export function ChatInput({
  onSend,
  disabled = false,
  isStreaming = false,
  isUploadingFiles = false,
  onStop,
  onEnqueue,
  placeholder = "Type your message...",
  className,
  inputId = "chat-input",
  droppedFiles,
  onDroppedFilesConsumed,
  hasSession = false,
}: Props) {
  const { copilotMode, setCopilotMode, isDryRun, setIsDryRun } =
    useCopilotUIStore();
  const showModeToggle = useGetFlag(Flag.CHAT_MODE_OPTION);
  const showDryRunToggle = showModeToggle;
  const [files, setFiles] = useState<File[]>([]);

  function handleToggleMode() {
    const next =
      copilotMode === "extended_thinking" ? "fast" : "extended_thinking";
    setCopilotMode(next);
    toast({
      title:
        next === "fast"
          ? "Switched to Fast mode"
          : "Switched to Extended Thinking mode",
      description:
        next === "fast"
          ? "Optimized for speed — ideal for simpler tasks."
          : "Responses may take longer.",
    });
  }

  function handleToggleDryRun() {
    const next = !isDryRun;
    setIsDryRun(next);
    toast({
      title: next ? "Test mode enabled" : "Test mode disabled",
      description: next
        ? "New chats will run agents in test mode."
        : "New chats will run agents normally.",
    });
  }

  // Merge files dropped onto the chat window into internal state.
  useEffect(() => {
    if (droppedFiles && droppedFiles.length > 0) {
      setFiles((prev) => [...prev, ...droppedFiles]);
      onDroppedFilesConsumed?.();
    }
  }, [droppedFiles, onDroppedFilesConsumed]);

  const hasFiles = files.length > 0;
  // isBusy disables non-essential interactions (attachment menu, voice recording)
  // but must not disable the textarea itself — streaming allows queued messages.
  const isBusy = disabled || isStreaming || isUploadingFiles;
  // The textarea is only truly disabled when the session is unavailable, not
  // during normal streaming (users can type and queue the next message).
  const isTextareaDisabled = disabled || isUploadingFiles;

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
    disabled: isTextareaDisabled,
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
    disabled: isTextareaDisabled,
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
            {showModeToggle && !isStreaming && (
              <ModeToggleButton
                mode={copilotMode}
                onToggle={handleToggleMode}
              />
            )}
            {showDryRunToggle && (!hasSession || isDryRun) && (
              <DryRunToggleButton
                isDryRun={isDryRun}
                isStreaming={isStreaming}
                readOnly={hasSession}
                onToggle={handleToggleDryRun}
              />
            )}
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
            {isStreaming && canSend && onEnqueue ? (
              <PromptInputButton
                tooltip="Queue message"
                onClick={() => {
                  if (value.trim()) {
                    void onEnqueue(value.trim());
                    setValue("");
                  }
                }}
                className="size-[2.625rem] rounded-full border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900"
              >
                <Tray className="size-4" weight="bold" />
              </PromptInputButton>
            ) : isStreaming ? (
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
