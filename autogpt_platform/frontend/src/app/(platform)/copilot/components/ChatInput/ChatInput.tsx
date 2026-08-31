import {
  PromptInputBody,
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
import { ComposerTray } from "./components/ComposerTray";
import { DryRunToggleButton } from "./components/DryRunToggleButton";
import { FileChips } from "./components/FileChips";
import { MentionDropdown } from "./components/MentionDropdown";
import { ModelToggleButton } from "./components/ModelToggleButton";
import { ModeToggleButton } from "./components/ModeToggleButton";
import { LLMRouteSelector } from "./components/LlmRouteSelector";
import { RecordingButton } from "./components/RecordingButton";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { WorkspaceFilePicker } from "./components/WorkspaceFilePicker/WorkspaceFilePicker";
import { useCopilotUIStore } from "../../store";
import { isTokenDevtoolEnabled } from "../../tokenDevtool/gate";
import { TokenDevtoolBadge } from "../TokenDevtoolBadge/TokenDevtoolBadge";
import { getFilesFromClipboard } from "./helpers";
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
  /** Session id for the dev-only token badge in the tray. */
  sessionId?: string | null;
  /** When true, the submit button is hidden until there is something to send. */
  hideSubmitWhenEmpty?: boolean;
  /** Recipient picker chip rendered before the mode chips (new-task state). */
  recipientPicker?: ReactNode;
}

// One text row is ~48px (24px line + 24px vertical padding); anything taller
// means the text wrapped and the composer stacks GPT-style: textarea full
// width on top, controls on their own row below.
const SINGLE_ROW_MAX_HEIGHT_PX = 56;

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
}: Props) {
  const {
    copilotChatMode,
    copilotModePinned,
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
  const [isMultiline, setIsMultiline] = useState(false);

  function handleToggleMode() {
    if (copilotModePinned) {
      toast({
        title: "Mode is locked while building an agent",
        description:
          "This session switched to Extended Thinking for agent building — building sessions stay on that engine.",
      });
      return;
    }
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

  // The brain-dump experience relocates the mode/model/dry-run chips into
  // the tray below the card; off keeps them as pills in the footer.
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);

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

  // Narrows to string, so neither render site needs to re-test sessionId.
  const devtoolSessionId = isTokenDevtoolEnabled() ? sessionId : null;
  const hasTrayItems =
    (showModeToggle && !isStreaming) ||
    (showDryRunToggle && !hasSession) ||
    Boolean(devtoolSessionId);

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
          "relative z-10 flex-col overflow-hidden !rounded-[1.75rem] border-zinc-200 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_20px_rgba(0,0,0,0.08)] has-[[data-slot=input-group-control]:focus-visible]:border-zinc-300 has-[[data-slot=input-group-control]:focus-visible]:ring-0",
          isRecording &&
            "border-red-400 ring-1 ring-red-400 has-[[data-slot=input-group-control]:focus-visible]:border-red-400 has-[[data-slot=input-group-control]:focus-visible]:ring-red-400",
        )}
      >
        <FileChips
          attachments={attachments}
          onRemove={handleRemoveAttachment}
          isUploading={isUploadingFiles}
        />
        <div className="flex w-full flex-wrap items-end">
          <InputGroupAddon
            align="inline-start"
            className="order-none gap-1 py-1 pl-1.5"
          >
            <ComposerPlusMenu
              onFilesSelected={handleFilesSelected}
              onUseWorkspaceFile={() => setIsPickerOpen(true)}
              onClearGuidedPrompt={handleClearGuidedPrompt}
              disabled={isBusy}
            />
            {recipientPicker}
            {!hasSession && <LLMRouteSelector />}
          </InputGroupAddon>
          <PromptInputBody
            className={cn(
              "relative block",
              isMultiline ? "order-first w-full" : "min-w-0 flex-1",
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
              onHeightChange={(height) =>
                setIsMultiline(height > SINGLE_ROW_MAX_HEIGHT_PX)
              }
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
          <InputGroupAddon
            align="inline-end"
            className="order-none ml-auto gap-1 py-1 pr-1.5"
          >
            {!isBrainDumpEnabled && showModeToggle && !isStreaming && (
              <>
                <ModeToggleButton
                  variant="pill"
                  mode={copilotChatMode}
                  onToggle={handleToggleMode}
                  pinned={copilotModePinned}
                />
                <ModelToggleButton
                  variant="pill"
                  model={copilotLlmModel}
                  onToggle={handleToggleModel}
                />
              </>
            )}
            {!isBrainDumpEnabled && showDryRunToggle && !hasSession && (
              <DryRunToggleButton
                variant="pill"
                isDryRun={isDryRun}
                onToggle={handleToggleDryRun}
              />
            )}
            {/* ComposerTray renders only under the brain-dump flag, so the
                badge is duplicated here to stay reachable in both layouts. */}
            {!isBrainDumpEnabled && devtoolSessionId && (
              <TokenDevtoolBadge sessionId={devtoolSessionId} />
            )}
            {showMicButton && (
              <RecordingButton
                isRecording={isRecording}
                isTranscribing={isTranscribing}
                isStreaming={isStreaming}
                disabled={disabled || isTranscribing || isStreaming}
                highlight={isMicGlowing}
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
              <PromptInputSubmit disabled={!canSend} />
            )}
          </InputGroupAddon>
        </div>

        <span id={`${inputId}-hint`} className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>
      </InputGroup>

      {/* Mode and model are per-message settings sent with each stream request,
          so they can be freely changed between turns in an existing session.
          Hide only while actively streaming (too late to change for that turn).
          DryRun is new-chat only: once a session exists its dry_run flag is
          locked and read from session metadata (sessionDryRun in useCopilotPage),
          with the banner in CopilotPage.tsx reflecting the actual state. */}
      {Boolean(isBrainDumpEnabled) && hasTrayItems && (
        <ComposerTray>
          {showModeToggle && !isStreaming && (
            <>
              <ModeToggleButton
                mode={copilotChatMode}
                onToggle={handleToggleMode}
                pinned={copilotModePinned}
              />
              <ModelToggleButton
                model={copilotLlmModel}
                onToggle={handleToggleModel}
              />
            </>
          )}
          {showDryRunToggle && !hasSession && (
            <DryRunToggleButton
              isDryRun={isDryRun}
              onToggle={handleToggleDryRun}
            />
          )}
          {devtoolSessionId && (
            <TokenDevtoolBadge
              sessionId={devtoolSessionId}
              className="ml-auto"
            />
          )}
        </ComposerTray>
      )}

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
