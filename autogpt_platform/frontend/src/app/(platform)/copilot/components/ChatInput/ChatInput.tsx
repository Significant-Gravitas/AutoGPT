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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { ChangeEvent, KeyboardEvent, useEffect, useState } from "react";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  type Attachment,
  type WorkspaceAttachment,
  partitionAttachments,
  workspaceItemToAttachment,
} from "../../helpers/workspaceAttachments";
import { AttachmentMenu } from "./components/AttachmentMenu";
import { BlockCaret } from "./components/BlockCaret";
import { DryRunToggleButton } from "./components/DryRunToggleButton";
import { FileChips } from "./components/FileChips";
import { MentionDropdown } from "./components/MentionDropdown";
import { ModelToggleButton } from "./components/ModelToggleButton";
import { ModeToggleButton } from "./components/ModeToggleButton";
import { RecordingButton } from "./components/RecordingButton";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { WorkspaceFilePicker } from "./components/WorkspaceFilePicker/WorkspaceFilePicker";
import { useCopilotUIStore } from "../../store";
import { useChatInput } from "./useChatInput";
import { useChatMentions } from "./useChatMentions";
import { useVoiceRecording } from "./useVoiceRecording";

interface Props {
  onSend: (
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) => void | Promise<void>;
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
  const showWorkspaceFiles = useGetFlag(Flag.CHAT_WORKSPACE_FILES);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isPickerOpen, setIsPickerOpen] = useState(false);

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
          : "Switched to Balanced model",
      description:
        next === "advanced"
          ? "Using the highest-capability model."
          : "Using the balanced default model.",
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
      setAttachments((prev) => [
        ...prev,
        ...droppedFiles.map((file) => ({ kind: "local" as const, file })),
      ]);
      onDroppedFilesConsumed?.();
    }
  }, [droppedFiles, onDroppedFilesConsumed]);

  const hasAttachments = attachments.length > 0;
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
      const { localFiles, workspaceFiles } = partitionAttachments(attachments);
      await onSend(
        message,
        localFiles.length > 0 ? localFiles : undefined,
        workspaceFiles.length > 0 ? workspaceFiles : undefined,
      );
      // Only clear after successful send (onSend throws on failure)
      setAttachments([]);
    },
    disabled: isTextareaDisabled,
    canSendEmpty: hasAttachments,
    inputId,
  });

  const mentions = useChatMentions({
    enabled: showWorkspaceFiles && !isBusy,
    value,
    setValue,
    addWorkspaceFile: handleWorkspaceFileSelected,
  });

  const [isEnqueueing, setIsEnqueueing] = useState(false);

  const {
    isRecording,
    isTranscribing,
    elapsedTime,
    toggleRecording,
    handleKeyDown: voiceHandleKeyDown,
    showMicButton,
    isInputDisabled,
    audioStream,
  } = useVoiceRecording({
    setValue,
    disabled: isTextareaDisabled,
    value,
    inputId,
    isStreaming,
  });

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    if (isRecording) return;
    baseHandleChange(e);
    mentions.detect(e.currentTarget);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (mentions.onKeyDown(e)) return;
    voiceHandleKeyDown(e);
  }

  const resolvedPlaceholder = isRecording
    ? ""
    : isTranscribing
      ? "Transcribing..."
      : placeholder;

  const canSend =
    !disabled &&
    (!!value.trim() || hasAttachments) &&
    !isRecording &&
    !isTranscribing;

  function handleFilesSelected(newFiles: File[]) {
    setAttachments((prev) => [
      ...prev,
      ...newFiles.map((file) => ({ kind: "local" as const, file })),
    ]);
  }

  function handleWorkspaceFileSelected(item: WorkspaceFileItem) {
    setAttachments((prev) => {
      if (prev.some((a) => a.kind === "workspace" && a.fileId === item.id)) {
        return prev;
      }
      return [...prev, workspaceItemToAttachment(item)];
    });
  }

  function handleWorkspaceFilesConfirmed(items: WorkspaceFileItem[]) {
    items.forEach(handleWorkspaceFileSelected);
  }

  function handleRemoveAttachment(index: number) {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  }

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      {mentions.isOpen && (
        <MentionDropdown
          files={mentions.files}
          isLoading={mentions.isLoading}
          isError={mentions.isError}
          highlightedIndex={mentions.highlightedIndex}
          highlightedRef={mentions.highlightedRef}
          onSelect={mentions.accept}
          onHighlight={mentions.setHighlightedIndex}
        />
      )}
      <InputGroup
        className={cn(
          "overflow-hidden border-zinc-200 has-[[data-slot=input-group-control]:focus-visible]:border-neutral-200 has-[[data-slot=input-group-control]:focus-visible]:ring-0",
          isRecording &&
            "border-red-400 ring-1 ring-red-400 has-[[data-slot=input-group-control]:focus-visible]:border-red-400 has-[[data-slot=input-group-control]:focus-visible]:ring-red-400",
        )}
      >
        <FileChips
          attachments={attachments}
          onRemove={handleRemoveAttachment}
          isUploading={isUploadingFiles}
        />
        <PromptInputBody className="relative block w-full">
          <PromptInputTextarea
            id={inputId}
            aria-label="Chat message input"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onBlur={mentions.close}
            disabled={isInputDisabled}
            placeholder={resolvedPlaceholder}
            className="caret-transparent placeholder:indent-3"
          />
          <BlockCaret textareaId={inputId} />
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
              onUseWorkspaceFile={() => setIsPickerOpen(true)}
              showWorkspaceOption={showWorkspaceFiles}
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
            {isStreaming && canSend && onEnqueue && (
              <PromptInputButton
                aria-label="Queue message"
                tooltip="Queue message"
                variant="default"
                disabled={isEnqueueing}
                onClick={async () => {
                  if (isEnqueueing) return;
                  const trimmed = value.trim();
                  if (trimmed) {
                    setIsEnqueueing(true);
                    try {
                      await onEnqueue(trimmed);
                      setValue("");
                    } finally {
                      setIsEnqueueing(false);
                    }
                  }
                }}
                className="size-[2.625rem] rounded-full border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900 disabled:border-zinc-200 disabled:bg-zinc-200 disabled:text-white disabled:opacity-100"
              >
                <ArrowUpIcon className="size-4" weight="bold" />
              </PromptInputButton>
            )}
            {isStreaming ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <PromptInputSubmit status="streaming" onStop={onStop} />
                </TooltipTrigger>
                <TooltipContent side="top">Stop</TooltipContent>
              </Tooltip>
            ) : (
              <PromptInputSubmit disabled={!canSend} />
            )}
          </div>
        </PromptInputFooter>
      </InputGroup>

      {showWorkspaceFiles && (
        <WorkspaceFilePicker
          isOpen={isPickerOpen}
          onClose={() => setIsPickerOpen(false)}
          onConfirm={handleWorkspaceFilesConfirmed}
        />
      )}
    </form>
  );
}
