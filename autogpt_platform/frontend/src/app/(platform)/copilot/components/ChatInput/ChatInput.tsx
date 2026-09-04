import {
  PromptInputButton,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { isGuidedPrompt } from "@/components/contextual/guidedPrompts";
import { toast } from "@/components/molecules/Toast/use-toast";
import { InputGroup, InputGroupAddon } from "@/components/ui/input-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  ChangeEvent,
  ClipboardEvent,
  KeyboardEvent,
  ReactNode,
  useEffect,
  useState,
} from "react";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  type Attachment,
  type WorkspaceAttachment,
  partitionAttachments,
  workspaceItemToAttachment,
} from "../../helpers/workspaceAttachments";
import { ComposerPlusMenu } from "./components/ComposerPlusMenu";
import { DryRunToggleButton } from "./components/DryRunToggleButton";
import { FileChips } from "./components/FileChips";
import { MentionDropdown } from "./components/MentionDropdown";
import { ConnectionPicker } from "./components/ConnectionPicker/ConnectionPicker";
import { RecordingButton } from "./components/RecordingButton";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { WorkspaceFilePicker } from "./components/WorkspaceFilePicker/WorkspaceFilePicker";
import { useCopilotUIStore } from "../../store";
import { isTokenDevtoolEnabled } from "../../tokenDevtool/gate";
import { TokenDevtoolBadge } from "../TokenDevtoolBadge/TokenDevtoolBadge";
import {
  CARD_ICON_BUTTON_CLASS,
  CARD_SEND_BUTTON_CLASS,
  getFilesFromClipboard,
} from "./helpers";
import { useChatInput } from "./useChatInput";
import { useChatMentions } from "./useChatMentions";
import { useOnboardingMicGlow } from "./useOnboardingMicGlow";
import { useVoiceRecording } from "./useVoiceRecording";
import { ArrowUp02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
  /** Session id for the dev-only token badge in the composer footer. */
  sessionId?: string | null;
  /** When true, the submit button is hidden until there is something to send. */
  hideSubmitWhenEmpty?: boolean;
  /** Recipient picker chip rendered before the mode chips (new-task state). */
  recipientPicker?: ReactNode;
  /** Card composer: the text always keeps its own row above the controls,
   *  instead of sharing a single pill row until it wraps. Empty state only. */
  stacked?: boolean;
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
  sessionId = null,
  hideSubmitWhenEmpty = false,
  recipientPicker,
  stacked = false,
}: Props) {
  const { isDryRun, setIsDryRun } = useCopilotUIStore();
  // Still the CHAT_MODE_OPTION flag, which no longer names what it gates: the
  // Fast/Thinking control it was created for is gone. Renaming it means an
  // LaunchDarkly change, so it is left until the combined picker restructures
  // these controls anyway. Splitting it now would need two new flags and would
  // hide both survivors until someone created them.
  const showAdvancedComposerControls = useGetFlag(Flag.CHAT_MODE_OPTION);
  const showWorkspaceFiles = useGetFlag(Flag.CHAT_WORKSPACE_FILES);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [isMultiline, setIsMultiline] = useState(false);

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
      // Chips clear eagerly for the same reason the text does (see
      // useChatInput.handleSend); a failed send restores them unless the
      // user already attached new ones in the meantime.
      const sent = attachments;
      setAttachments([]);
      try {
        await onSend(
          message,
          localFiles.length > 0 ? localFiles : undefined,
          workspaceFiles.length > 0 ? workspaceFiles : undefined,
        );
      } catch (error) {
        setAttachments((prev) => (prev.length > 0 ? prev : sent));
        throw error;
      }
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

  const { isGlowing: isMicGlowing, dismissGlow } = useOnboardingMicGlow({
    isTranscribing,
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

  function handlePaste(e: ClipboardEvent<HTMLTextAreaElement>) {
    if (isBusy) return;
    const files = getFilesFromClipboard(e.clipboardData);
    if (files.length === 0) return;
    e.preventDefault();
    handleFilesSelected(files);
  }

  const resolvedPlaceholder = isRecording
    ? ""
    : isTranscribing
      ? "Transcribing..."
      : placeholder;

  // Narrows to string, so the render site needn't re-test sessionId.
  const devtoolSessionId = isTokenDevtoolEnabled() ? sessionId : null;
  const canSend =
    !disabled &&
    (!!value.trim() || hasAttachments) &&
    !isRecording &&
    !isTranscribing;

  function handleClearGuidedPrompt() {
    // Only discard untouched guided prompts — never a draft the user typed
    // or edited themselves.
    if (isGuidedPrompt(value)) setValue("");
  }

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
      {/* GPT-style composer: a single pill row — plus on the left, textarea
          in the middle, quiet toggles + mic + send on the right. items-end
          keeps the controls pinned to the bottom edge as the textarea grows. */}
      <InputGroup
        className={cn(
          "relative z-10 flex-col overflow-hidden !rounded-[2rem] border-zinc-200 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_20px_rgba(0,0,0,0.08)] has-[[data-slot=input-group-control]:focus-visible]:border-zinc-300 has-[[data-slot=input-group-control]:focus-visible]:ring-0",
          // Card composer: a hairline ring and a shallow drop instead of the
          // pill's deep shadow, so it reads as a surface the text sits on.
          stacked &&
            "gap-3 !rounded-3xl border-transparent px-3.5 pb-3.5 pt-3 shadow-[0_0_0_0.5px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02)] has-[[data-slot=input-group-control]:focus-visible]:border-transparent",
          isRecording &&
            "border-red-400 ring-1 ring-red-400 has-[[data-slot=input-group-control]:focus-visible]:border-red-400 has-[[data-slot=input-group-control]:focus-visible]:ring-red-400",
        )}
      >
        <FileChips
          attachments={attachments}
          onRemove={handleRemoveAttachment}
          isUploading={isUploadingFiles}
          stacked={stacked}
        />
        <div
          className={cn(
            "flex w-full flex-wrap",
            stacked ? "items-center" : "items-end",
          )}
        >
          <InputGroupAddon
            align="inline-start"
            className={cn(
              "order-none gap-1 py-1 pl-1.5",
              stacked && "gap-1.5 p-0",
            )}
          >
            <ComposerPlusMenu
              onFilesSelected={handleFilesSelected}
              onUseWorkspaceFile={() => setIsPickerOpen(true)}
              onClearGuidedPrompt={handleClearGuidedPrompt}
              disabled={isBusy}
              className={
                stacked
                  ? cn(
                      CARD_ICON_BUTTON_CLASS,
                      "[&[aria-expanded=true]_svg]:rotate-45 [&_svg]:transition-transform [&_svg]:duration-200",
                    )
                  : undefined
              }
            />
            {recipientPicker}
          </InputGroupAddon>
          {/* Must be a real flex item: `order`/`w-full` are ignored on a
              `display: contents` box, which is what PromptInputBody was. */}
          <div
            className={cn(
              "relative",
              stacked || isMultiline ? "order-first w-full" : "min-w-0 flex-1",
            )}
          >
            <PromptInputTextarea
              id={inputId}
              aria-label="Chat message input"
              value={value}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              onBlur={mentions.close}
              disabled={isInputDisabled}
              placeholder={resolvedPlaceholder}
              onMultilineChange={setIsMultiline}
              className={stacked ? "px-0.5 py-1" : undefined}
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
          <InputGroupAddon
            align="inline-end"
            className={cn(
              "order-none ml-auto gap-1 py-1 pr-1.5",
              stacked && "gap-1.5 p-0",
            )}
          >
            {/* Connection and tier are per-message settings, so they remain
                changeable between turns in an existing session. The card
                composer leaves this to the page's top-right control. */}
            {!stacked && (!hasSession || !isStreaming) && (
              <ConnectionPicker connectionLocked={hasSession} />
            )}
            {showAdvancedComposerControls && !hasSession && (
              <DryRunToggleButton
                isDryRun={isDryRun}
                onToggle={handleToggleDryRun}
              />
            )}
            {devtoolSessionId && (
              <TokenDevtoolBadge sessionId={devtoolSessionId} />
            )}
            {showMicButton && (
              <RecordingButton
                isRecording={isRecording}
                isTranscribing={isTranscribing}
                isStreaming={isStreaming}
                disabled={disabled || isTranscribing || isStreaming}
                highlight={isMicGlowing}
                className={stacked ? CARD_ICON_BUTTON_CLASS : undefined}
                onClick={() => {
                  dismissGlow();
                  toggleRecording();
                }}
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
                <Icon icon={ArrowUp02Icon} className="size-4" />
              </PromptInputButton>
            )}
            {isStreaming ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <PromptInputSubmit status="streaming" onStop={onStop} />
                </TooltipTrigger>
                <TooltipContent side="top">Stop</TooltipContent>
              </Tooltip>
            ) : hideSubmitWhenEmpty && !canSend ? null : (
              <PromptInputSubmit
                disabled={!canSend}
                className={stacked ? CARD_SEND_BUTTON_CLASS : undefined}
              />
            )}
          </InputGroupAddon>
        </div>

        <span id={`${inputId}-hint`} className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>
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
