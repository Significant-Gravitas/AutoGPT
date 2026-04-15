import {
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { toast } from "@/components/molecules/Toast/use-toast";
import { InputGroup } from "@/components/ui/input-group";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ChangeEvent, useEffect, useState } from "react";
import { AttachmentMenu } from "./components/AttachmentMenu";
import { DryRunToggleButton } from "./components/DryRunToggleButton";
import { FileChips } from "./components/FileChips";
import { ModelToggleButton } from "./components/ModelToggleButton";
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
  placeholder = "Type your message...",
  className,
  inputId = "chat-input",
  droppedFiles,
  onDroppedFilesConsumed,
  hasSession = false,
}: Props) {
  const {
    copilotChatMode,
    setCopilotChatMode,
    copilotLlmModel,
    setCopilotLlmModel,
    isDryRun,
    setIsDryRun,
  } = useCopilotUIStore();
  const showModeToggle = useGetFlag(Flag.CHAT_MODE_OPTION);
  const showDryRunToggle = showModeToggle;
  const [files, setFiles] = useState<File[]>([]);

  function handleToggleMode() {
    const next =
      copilotChatMode === "extended_thinking" ? "fast" : "extended_thinking";
    setCopilotChatMode(next);
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

  function handleToggleModel() {
    const next = copilotLlmModel === "advanced" ? "standard" : "advanced";
    setCopilotLlmModel(next);
    toast({
      title:
        next === "advanced"
          ? "Switched to Advanced model"
          : "Switched to Standard model",
      description:
        next === "advanced"
          ? "Using the highest-capability model."
          : "Using the balanced standard model.",
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
            {/* Mode and model are per-message settings sent with each stream request,
                so they can be freely changed between turns in an existing session.
                Hide only while actively streaming (too late to change for that turn). */}
            {showModeToggle && !isStreaming && (
              <ModeToggleButton
                mode={copilotChatMode}
                onToggle={handleToggleMode}
              />
            )}
            {showModeToggle && !isStreaming && (
              <ModelToggleButton
                model={copilotLlmModel}
                onToggle={handleToggleModel}
              />
            )}
            {/* DryRun button only on new chats: once a session exists its
                dry_run flag is locked and should be read from session metadata
                (sessionDryRun in useCopilotPage), not toggled here. The banner
                in CopilotPage.tsx reflects the actual session state. */}
            {showDryRunToggle && !hasSession && (
              <DryRunToggleButton
                isDryRun={isDryRun}
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
