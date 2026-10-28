import { CredentialMentionEditor } from "../CredentialMention/CredentialMentionEditor";
import type { MentionInput } from "./useChatMentions";
import {
  PromptInputButton,
  PromptInputSubmit,
} from "@/components/organisms/PromptInput/PromptInput";
import { isGuidedPrompt } from "@/components/contextual/guidedPrompts";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  InputGroup,
  InputGroupAddon,
} from "@/components/molecules/InputGroup/InputGroup";
import { containsLikelySecret } from "@/lib/secret-detection";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  ClipboardEvent,
  KeyboardEvent,
  ReactNode,
  memo,
  useEffect,
  useState,
} from "react";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  type Attachment,
  type WorkspaceAttachment,
  MAX_FOLDER_ATTACHMENTS,
  appendWithinCap,
  MAX_ATTACHMENTS,
  partitionAttachments,
  workspaceFolderToAttachment,
  workspaceItemToAttachment,
} from "../../helpers/workspaceAttachments";
import type { PickedItem } from "./components/WorkspaceFilePicker/useWorkspaceFilePicker";
import { AtResetFilesNotice } from "./components/AtResetFilesNotice";
import { AttachmentCapNotice } from "./components/AttachmentCapNotice";
import { ComposerOverflowMenu } from "./components/ComposerOverflowMenu";
import {
  ComposerPlusMenu,
  type ComposerMenuItem,
} from "./components/ComposerPlusMenu";
import { DryRunToggleButton } from "./components/DryRunToggleButton";
import { FileChips } from "./components/FileChips";
import { MentionConnectDialog } from "./components/MentionConnectDialog";
import { MentionDropdown } from "./components/MentionDropdown";
import { ConnectionPickerDialog } from "./components/ConnectionPicker/ConnectionPickerDialog";
import { NoAIConnectionNotice } from "./components/NoAIConnectionNotice/NoAIConnectionNotice";
import { RecordingButton } from "./components/RecordingButton";
import { RecordingIndicator } from "./components/RecordingIndicator";
import { SecretPasteNotice } from "./components/SecretPasteNotice";
import { StopButton } from "./components/StopButton";
import { TranscriptionErrorBar } from "./components/TranscriptionErrorBar";
import { WorkspaceFilePicker } from "./components/WorkspaceFilePicker/WorkspaceFilePicker";
import { useCopilotUIStore } from "../../store";
import { isTokenDevtoolEnabled } from "../../tokenDevtool/gate";
import { TokenDevtoolDialog } from "../TokenDevtoolBadge/TokenDevtoolDialog";
import {
  sendAtResetFailureTitle,
  SEND_AT_RESET_LABEL,
} from "../SendAtReset/helpers";
import {
  CARD_ICON_BUTTON_CLASS,
  CARD_SEND_BUTTON_CLASS,
  COMPACT_ICON_BUTTON_CLASS,
  COMPACT_SEND_BUTTON_CLASS,
  MOBILE_SEND_BUTTON_CLASS,
  getFilesFromClipboard,
} from "./helpers";
import { useChatInput } from "./useChatInput";
import { useChatMentions } from "./useChatMentions";
import { useConnectedIntegrations } from "./useConnectedIntegrations";
import { useOnboardingMicGlow } from "./useOnboardingMicGlow";
import { useVoiceRecording } from "./useVoiceRecording";
import {
  ArrowUp02Icon,
  DashboardSpeed02Icon,
  FlashIcon,
  VoiceIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  onSend: (
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) => void | Promise<void>;
  disabled?: boolean;
  sendDisabled?: boolean;
  isStreaming?: boolean;
  /** The sent message's turn has not started: a Stop now would cancel nothing
   *  on the server and drop the message, so the button only shows progress. */
  isTurnStarting?: boolean;
  isUploadingFiles?: boolean;
  onStop?: () => void;
  /** Called to enqueue a message when copilot is streaming and user has typed text. */
  onEnqueue?: (message: string) => void | Promise<void>;
  placeholder?: string;
  className?: string;
  inputId?: string;
  draftKey?: string;
  /** Off leaves keyboard focus where it is, on mount and when re-enabled. */
  autoFocus?: boolean;
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
  /** Starts voice mode from the + menu. Absent when the flag is off. */
  voiceMode?: { onStart: () => void; disabled?: boolean };
  /** The chat's approval-mode selector. Absent when the flag is off. */
  modeSelector?: ReactNode;
  /**
   * Replaces the composer's controls while voice mode is on: typing,
   * attachments and send do nothing hands-free, and a bar of its own above
   * the composer covered the last message's buttons.
   */
  voiceBar?: ReactNode;
  /** Compact composer for side panels: tighter radius, flat shadow, smaller
   *  controls, and no per-message connection chip. */
  variant?: "default" | "compact";
  /** Expert the chat is scoped to. Workspace-file suggestions and the picker
   *  then only offer files from that expert's conversations. */
  expertId?: string | null;
  /** Names that expert in the picker's filter row. */
  expertName?: string | null;
  /** Who the message goes to, named in the secret warning: the expert, or
   *  Otto when unset. */
  assistantName?: string | null;
  /** Attachments stay open during a reply, and a message with attachments
   *  goes to `onSend`, which must hold it until the reply ends. */
  canAttachWhileStreaming?: boolean;
  /** Out of AutoPilot budget with a reset ahead: the text stays editable and
   *  submitting sets it to send at the reset. Text only. */
  atReset?: { onSchedule: (message: string) => Promise<void> };
}

// memo-boundary: the chat page re-renders on every streamed flush, and the
// composer's props stay the same while the reply streams in. The builder,
// memory and team chats render it from folders the compiler does not cover,
// so their parents do not cache its element.
export const ChatInput = memo(function ChatInput({
  onSend,
  disabled = false,
  sendDisabled = false,
  isStreaming = false,
  isTurnStarting = false,
  isUploadingFiles = false,
  onStop,
  onEnqueue,
  placeholder = "Type your message...",
  className,
  inputId = "chat-input",
  draftKey,
  autoFocus = true,
  droppedFiles,
  onDroppedFilesConsumed,
  hasSession = false,
  sessionId = null,
  hideSubmitWhenEmpty = false,
  recipientPicker,
  stacked = false,
  voiceMode,
  modeSelector,
  voiceBar,
  variant = "default",
  expertId = null,
  expertName = null,
  assistantName = null,
  canAttachWhileStreaming = false,
  atReset,
}: Props) {
  const isDryRun = useCopilotUIStore((s) => s.isDryRun);
  const setIsDryRun = useCopilotUIStore((s) => s.setIsDryRun);
  // Still the CHAT_MODE_OPTION flag, which no longer names what it gates: the
  // Fast/Thinking control it was created for is gone. Renaming it means an
  // LaunchDarkly change, so it is left until the combined picker restructures
  // these controls anyway. Splitting it now would need two new flags and would
  // hide both survivors until someone created them.
  const showAdvancedComposerControls = useGetFlag(Flag.CHAT_MODE_OPTION);
  const showWorkspaceFiles = useGetFlag(Flag.CHAT_WORKSPACE_FILES);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  // How many files the cap turned away on the last attach; 0 hides the notice.
  const [refusedCount, setRefusedCount] = useState(0);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [isMultiline, setIsMultiline] = useState(false);

  function handleToggleDryRun() {
    const next = !isDryRun;
    setIsDryRun(next);
    toast({
      title: next ? "Test mode enabled" : "Test mode disabled",
      description: next
        ? "New chats will run workflows in test mode."
        : "New chats will run workflows normally.",
    });
  }

  function addAttachments(incoming: Attachment[]) {
    // Outside the updater: React re-invokes an updater (twice under
    // StrictMode), which would toast the refusal more than once.
    const { next, refused, refusedFolders } = appendWithinCap(
      attachments,
      incoming,
    );
    setAttachments(next);
    setRefusedCount(refused);
    if (refusedFolders > 0) {
      toast({
        title: `Up to ${MAX_FOLDER_ATTACHMENTS} folders per message`,
        description: `${refusedFolders} not added.`,
      });
    }
  }

  function handleFilesSelected(newFiles: File[]) {
    addAttachments(newFiles.map(toLocalAttachment));
  }

  function handleWorkspaceFileSelected(item: WorkspaceFileItem) {
    addAttachments([workspaceItemToAttachment(item)]);
  }

  // Merge files dropped onto the chat window into internal state.
  useEffect(() => {
    if (droppedFiles && droppedFiles.length > 0) {
      addAttachments(droppedFiles.map(toLocalAttachment));
      onDroppedFilesConsumed?.();
    }
  }, [droppedFiles, onDroppedFilesConsumed]);

  const hasAttachments = attachments.length > 0;
  const isAtCap = attachments.length >= MAX_ATTACHMENTS;
  // isBusy disables non-essential interactions (attachment menu, voice recording)
  // but must not disable the textarea itself — streaming allows queued messages.
  const isBusy =
    disabled || isUploadingFiles || (isStreaming && !canAttachWhileStreaming);
  // The textarea is only truly disabled when the session is unavailable, not
  // during normal streaming (users can type and queue the next message).
  const isTextareaDisabled = disabled || isUploadingFiles;

  async function sendWithAttachments(message: string) {
    const { localFiles, workspaceAttachments } =
      partitionAttachments(attachments);
    // Chips clear eagerly for the same reason the text does (see
    // useChatInput.handleSend); a failed send restores them unless the
    // user already attached new ones in the meantime.
    const sent = attachments;
    setAttachments([]);
    setRefusedCount(0);
    const files = localFiles.length > 0 ? localFiles : undefined;
    const workspaceFiles =
      workspaceAttachments.length > 0 ? workspaceAttachments : undefined;
    try {
      await onSend(message, files, workspaceFiles);
    } catch (error) {
      setAttachments((prev) => (prev.length > 0 ? prev : sent));
      throw error;
    }
  }

  const { value, setValue, handleSubmit, handleEnqueue, isEnqueueing } =
    useChatInput({
      onSend: atReset?.onSchedule ?? sendWithAttachments,
      onEnqueue,
      isStreaming,
      hasAttachments,
      // A scheduled send carries text only, so attachments hold it back.
      disabled: isTextareaDisabled || (atReset ? hasAttachments : sendDisabled),
      canSendEmpty: hasAttachments && !atReset,
      inputId,
      draftKey,
      autoFocus,
      failureTitle: atReset ? sendAtResetFailureTitle : undefined,
    });

  const integrations = useConnectedIntegrations(expertId);

  const mentions = useChatMentions({
    enabled: !isBusy,
    value,
    setValue,
    addWorkspaceFile: handleWorkspaceFileSelected,
    addWorkspaceFolder: (folder, subfolderCount) =>
      addAttachments([workspaceFolderToAttachment(folder, subfolderCount)]),
    expertId,
    // Files and folders become attachments, so the cap closes them off;
    // integrations only edit the text and stay available.
    includeWorkspaceFiles: showWorkspaceFiles && !isAtCap,
    integrations,
  });

  const {
    isRecording,
    isTranscribing,
    transcriptionError,
    hasFailedRecording,
    retryTranscription,
    downloadFailedRecording,
    dismissTranscriptionError,
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

  function handleChange(nextValue: string, input: MentionInput) {
    if (isRecording) return;
    setValue(nextValue);
    mentions.detect(input);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLElement>) {
    if (mentions.onKeyDown(e)) return;
    voiceHandleKeyDown(e);
  }

  function handlePaste(e: ClipboardEvent<HTMLElement>) {
    if (isBusy || atReset) return;
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
    (atReset
      ? !!value.trim() && !hasAttachments
      : !sendDisabled && (!!value.trim() || hasAttachments)) &&
    !isRecording &&
    !isTranscribing;
  // The pending endpoint takes text only, so attachments are queued through
  // `onSend`, and only by a host that holds them until the reply ends.
  const canQueue = hasAttachments ? canAttachWhileStreaming : !!onEnqueue;

  function handleClearGuidedPrompt() {
    // Only discard untouched guided prompts — never a draft the user typed
    // or edited themselves.
    if (isGuidedPrompt(value)) setValue("");
  }

  function handlePickerConfirmed(items: PickedItem[]) {
    addAttachments(
      items.map((item) =>
        item.kind === "folder"
          ? workspaceFolderToAttachment(item.folder, item.subfolderCount)
          : workspaceItemToAttachment(item.file),
      ),
    );
  }

  function handleRemoveAttachment(index: number) {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
    // Removing one frees a slot, so the count the notice quotes is now stale.
    setRefusedCount(0);
  }

  const isCompact = variant === "compact";
  // Voice mode, model tier and the dev token readout sit behind the +
  // instead of as icons in the tray; the last two open in a dialog.
  const [plusPanel, setPlusPanel] = useState<"model" | "devtool" | null>(null);
  const showModelControl =
    !stacked && !isCompact && (!hasSession || !isStreaming);
  if (!showModelControl && plusPanel === "model") {
    setPlusPanel(null);
  }
  const plusMenuItems: ComposerMenuItem[] = [
    ...(voiceMode
      ? [
          {
            key: "voice",
            label: "Voice mode",
            icon: VoiceIcon,
            disabled: voiceMode.disabled,
            onSelect: voiceMode.onStart,
          },
        ]
      : []),
    ...(showModelControl
      ? [
          {
            key: "model",
            label: "Model and connection",
            icon: FlashIcon,
            onSelect: () => setPlusPanel("model"),
          },
        ]
      : []),
    ...(devtoolSessionId
      ? [
          {
            key: "devtool",
            label: "Token usage",
            icon: DashboardSpeed02Icon,
            onSelect: () => setPlusPanel("devtool"),
          },
        ]
      : []),
  ];
  const iconButtonClass = stacked
    ? CARD_ICON_BUTTON_CLASS
    : isCompact
      ? COMPACT_ICON_BUTTON_CLASS
      : undefined;
  const sendButtonClass = stacked
    ? CARD_SEND_BUTTON_CLASS
    : isCompact
      ? COMPACT_SEND_BUTTON_CLASS
      : MOBILE_SEND_BUTTON_CLASS;

  // Per-message settings sit behind the "..." button, so the row stays
  // usable at phone width.
  const showDryRunToggle = showAdvancedComposerControls && !hasSession;
  const hasOverflowControls = !!modeSelector || showDryRunToggle;

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      {mentions.isOpen && (
        <MentionDropdown
          options={mentions.options}
          showFiles={mentions.showFiles}
          hasIntegrations={mentions.hasIntegrations}
          isLoading={mentions.isLoading}
          isError={mentions.isError}
          highlightedIndex={mentions.highlightedIndex}
          highlightedRef={mentions.highlightedRef}
          onSelect={mentions.accept}
          onHighlight={mentions.setHighlightedIndex}
        />
      )}
      <NoAIConnectionNotice enabled={!hasSession} />
      {/* GPT-style composer: a single pill row — plus on the left, textarea
          in the middle, quiet toggles + mic + send on the right. items-end
          keeps the controls pinned to the bottom edge as the textarea grows. */}
      <InputGroup
        className={cn(
          // TODO(tokens): no exact role for shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_20px_rgba(0,0,0,0.08)]
          "relative z-10 flex-col overflow-hidden !rounded-[2rem] border-border-base shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_20px_rgba(0,0,0,0.08)] has-[[data-slot=input-group-control]:focus-visible]:border-border-strong has-[[data-slot=input-group-control]:focus-visible]:ring-0 max-md:!rounded-3xl",
          // Card composer: a hairline border and a shallow drop instead of
          // the pill's deep shadow, so it reads as a surface the text sits on.
          stacked &&
            // TODO(tokens): no exact role for shadow-[0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02)]
            "gap-3 !rounded-3xl border-border-base px-3.5 pb-3.5 pt-3 shadow-[0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02)] has-[[data-slot=input-group-control]:focus-visible]:border-border-strong",
          isCompact &&
            // TODO(tokens): no exact role for shadow-[0_1px_2px_rgba(0,0,0,0.04)]; no fitting role for has-[[data-slot=input-group-control]:focus-visible]:border-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
            "!rounded-xl border-border-base shadow-[0_1px_2px_rgba(0,0,0,0.04)] has-[[data-slot=input-group-control]:focus-visible]:border-zinc-400",
          isRecording &&
            // TODO(tokens): no exact role for border-red-400, ring-red-400, has-[[data-slot=input-group-control]:focus-visible]:border-red-400, has-[[data-slot=input-group-control]:focus-visible]:ring-red-400
            "border-red-400 ring-1 ring-red-400 has-[[data-slot=input-group-control]:focus-visible]:border-red-400 has-[[data-slot=input-group-control]:focus-visible]:ring-red-400",
        )}
      >
        {voiceBar}
        {!voiceBar && transcriptionError && hasFailedRecording && (
          <TranscriptionErrorBar
            message={transcriptionError}
            isRetrying={isTranscribing}
            onRetry={retryTranscription}
            onDownload={downloadFailedRecording}
            onDismiss={dismissTranscriptionError}
            className={stacked ? undefined : "mt-1.5"}
          />
        )}
        {refusedCount > 0 && (
          <AttachmentCapNotice
            refusedCount={refusedCount}
            onDismiss={() => setRefusedCount(0)}
            className={stacked ? undefined : "mt-1.5"}
          />
        )}
        {atReset && hasAttachments && (
          <AtResetFilesNotice className={stacked ? undefined : "mt-1.5"} />
        )}
        <SecretPasteNotice
          visible={containsLikelySecret(value)}
          assistantName={assistantName ?? "Otto"}
          className={stacked ? undefined : "mt-1.5"}
        />
        <FileChips
          attachments={attachments}
          onRemove={handleRemoveAttachment}
          isUploading={isUploadingFiles}
          stacked={stacked}
        />
        <div
          className={cn(
            "flex w-full flex-wrap",
            stacked || isCompact ? "items-center" : "items-end",
            // tailwind-merge drops `flex` for `hidden`: the draft and the
            // attachments survive the round trip through voice mode.
            voiceBar && "hidden",
          )}
        >
          <InputGroupAddon
            align="inline-start"
            className={cn(
              "relative order-none gap-1 py-1 pl-1.5",
              stacked && "gap-1.5 p-0",
              isCompact && "gap-0.5 py-1 pl-1",
            )}
          >
            <ComposerPlusMenu
              onFilesSelected={handleFilesSelected}
              onUseWorkspaceFile={() => setIsPickerOpen(true)}
              onClearGuidedPrompt={handleClearGuidedPrompt}
              disabled={isBusy || !!atReset}
              isAtCap={isAtCap}
              extraItems={plusMenuItems}
              className={
                stacked
                  ? cn(
                      CARD_ICON_BUTTON_CLASS,
                      "[&[aria-expanded=true]_svg]:rotate-45 [&_svg]:transition-transform [&_svg]:duration-200",
                    )
                  : iconButtonClass
              }
            />
            {recipientPicker}
          </InputGroupAddon>
          {/* Must be a real flex item: `order`/`w-full` are ignored on a
              `display: contents` box, which is what PromptInputBody was. */}
          <div
            className={cn(
              "relative",
              stacked || isMultiline
                ? "order-first w-full"
                : "min-w-0 flex-1 max-md:order-first max-md:w-full max-md:basis-full",
            )}
          >
            <CredentialMentionEditor
              id={inputId}
              value={value}
              onInputReady={mentions.bindInput}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              onBlur={mentions.close}
              disabled={isInputDisabled}
              placeholder={resolvedPlaceholder}
              onMultilineChange={setIsMultiline}
              className={cn(
                "max-md:max-h-32",
                stacked && "px-0.5 py-1",
                isCompact && "text-body leading-5 md:text-body md:leading-5",
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
          <InputGroupAddon
            align="inline-end"
            className={cn(
              "order-none ml-auto gap-1 py-1 pr-1.5",
              stacked && "gap-1.5 p-0",
              isCompact && "gap-0.5 py-1 pr-1",
            )}
          >
            {hasOverflowControls && (
              <ComposerOverflowMenu
                forceClosed={!!voiceBar}
                className={iconButtonClass}
              >
                {modeSelector}
                {showDryRunToggle && (
                  <DryRunToggleButton
                    isDryRun={isDryRun}
                    onToggle={handleToggleDryRun}
                  />
                )}
              </ComposerOverflowMenu>
            )}
            {showMicButton && (
              <RecordingButton
                isRecording={isRecording}
                isTranscribing={isTranscribing}
                isStreaming={isStreaming}
                disabled={disabled || isTranscribing || isStreaming}
                highlight={isMicGlowing}
                className={iconButtonClass}
                onClick={() => {
                  dismissGlow();
                  toggleRecording();
                }}
              />
            )}
            {isStreaming && canSend && canQueue && (
              <PromptInputButton
                aria-label="Queue message"
                tooltip="Queue message"
                variant="default"
                disabled={isEnqueueing}
                type={hasAttachments ? "submit" : "button"}
                onClick={
                  hasAttachments ? undefined : () => void handleEnqueue()
                }
                className={cn(
                  "size-[2.625rem] rounded-full border-action bg-action text-action-foreground hover:border-action-hover hover:bg-action-hover disabled:border-border-base disabled:bg-surface-active disabled:text-action-foreground disabled:opacity-100",
                  sendButtonClass,
                )}
              >
                <Icon icon={ArrowUp02Icon} className="size-4" />
              </PromptInputButton>
            )}
            {isStreaming && isTurnStarting ? (
              <PromptInputSubmit
                status="submitted"
                aria-label="Sending"
                className={sendButtonClass}
              />
            ) : isStreaming ? (
              <StopButton onStop={onStop} className={sendButtonClass} />
            ) : atReset ? (
              <PromptInputSubmit
                disabled={!canSend}
                aria-label={SEND_AT_RESET_LABEL}
                className={cn(sendButtonClass, "!w-auto px-4 font-medium")}
              >
                {SEND_AT_RESET_LABEL}
              </PromptInputSubmit>
            ) : hideSubmitWhenEmpty && !canSend ? null : (
              <PromptInputSubmit
                disabled={!canSend}
                className={sendButtonClass}
              />
            )}
          </InputGroupAddon>
        </div>

        <span id={`${inputId}-hint`} className="sr-only">
          Press Enter to send, Shift+Enter for new line, Space to record voice
        </span>
      </InputGroup>
      {showModelControl && (
        <ConnectionPickerDialog
          connectionLocked={hasSession}
          open={plusPanel === "model"}
          onOpenChange={(open) => setPlusPanel(open ? "model" : null)}
        />
      )}
      {devtoolSessionId && (
        <TokenDevtoolDialog
          sessionId={devtoolSessionId}
          open={plusPanel === "devtool"}
          onOpenChange={(open) => setPlusPanel(open ? "devtool" : null)}
        />
      )}
      {showWorkspaceFiles && (
        <WorkspaceFilePicker
          key={expertId ?? "everyone"}
          isOpen={isPickerOpen}
          onClose={() => setIsPickerOpen(false)}
          onConfirm={handlePickerConfirmed}
          expertId={expertId}
          expertName={expertName}
        />
      )}
      <MentionConnectDialog
        request={mentions.connectRequest}
        expertId={expertId}
        expertName={expertName}
        onCancel={mentions.cancelConnect}
        onConnected={mentions.resumeAfterConnect}
      />
    </form>
  );
});

function toLocalAttachment(file: File): Attachment {
  return { kind: "local", file };
}
