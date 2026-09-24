import { createPortal } from "react-dom";
import {
  useId,
  useLayoutEffect,
  useRef,
  type ClipboardEvent,
  type KeyboardEvent,
} from "react";
import { PromptInputTextarea } from "@/components/ai-elements/prompt-input";
import { parseCredentialMentions } from "./helpers";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import type { MentionInput } from "../ChatInput/useChatMentions";
import { CredentialMentionBadge } from "./CredentialMentionBadge";
import { mentionInputFor, readMentionEditor } from "./editorHelpers";
import { useCredentialMentionEditor } from "./useCredentialMentionEditor";

interface Props {
  id?: string;
  value: string;
  onChange: (value: string, input: MentionInput) => void;
  onKeyDown: (event: KeyboardEvent<HTMLElement>) => void;
  onPaste?: (event: ClipboardEvent<HTMLElement>) => void;
  onBlur?: () => void;
  onInputReady?: (input: MentionInput) => void;
  onMultilineChange?: (multiline: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function CredentialMentionEditor({
  id,
  value,
  onChange,
  onKeyDown,
  onPaste,
  onBlur,
  onInputReady,
  onMultilineChange,
  disabled,
  placeholder,
  className,
}: Props) {
  const fallbackId = useId();
  const inputId = id ?? fallbackId;
  const rich = parseCredentialMentions(value).some(
    (part) => typeof part !== "string",
  );
  const wasRich = useRef(rich);
  const { editorRef, badges } = useCredentialMentionEditor(
    value,
    onMultilineChange,
  );

  useLayoutEffect(() => {
    const input = document.getElementById(inputId);
    if (input && wasRich.current !== rich) input.focus();
    wasRich.current = rich;
    if (rich && editorRef.current)
      onInputReady?.(mentionInputFor(editorRef.current));
  });

  function handleInput() {
    const editor = editorRef.current;
    if (editor) onChange(readMentionEditor(editor), mentionInputFor(editor));
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    onKeyDown(event);
    if (event.defaultPrevented || !isKey(event, "Enter")) return;
    event.preventDefault();
    if (event.shiftKey) document.execCommand("insertText", false, "\n");
    else event.currentTarget.closest("form")?.requestSubmit();
  }

  function handlePaste(event: ClipboardEvent<HTMLDivElement>) {
    onPaste?.(event);
    if (event.defaultPrevented) return;
    event.preventDefault();
    document.execCommand(
      "insertText",
      false,
      event.clipboardData.getData("text/plain"),
    );
  }

  if (!rich)
    return (
      <PromptInputTextarea
        id={inputId}
        aria-label="Chat message input"
        value={value}
        onChange={(event) =>
          onChange(event.currentTarget.value, event.currentTarget)
        }
        onKeyDown={onKeyDown}
        onPaste={onPaste}
        onBlur={onBlur}
        onMultilineChange={onMultilineChange}
        disabled={disabled}
        placeholder={placeholder}
        className={className}
      />
    );

  return (
    <>
      <input type="hidden" name="message" value={value} />
      <div
        ref={editorRef}
        id={inputId}
        role="textbox"
        aria-label="Chat message input"
        aria-multiline="true"
        aria-disabled={disabled}
        data-slot="input-group-control"
        data-placeholder={placeholder}
        contentEditable={!disabled}
        suppressContentEditableWarning
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        onBlur={onBlur}
        className={cn(
          "max-h-48 min-h-10 w-full overflow-y-auto whitespace-pre-wrap break-words bg-transparent px-3 py-2 text-base leading-7 outline-none empty:before:pointer-events-none empty:before:text-zinc-400 empty:before:content-[attr(data-placeholder)]",
          disabled && "cursor-not-allowed opacity-50",
          className,
        )}
      />
      {badges.map(({ element, mention }, index) =>
        createPortal(
          <CredentialMentionBadge
            name={mention.name}
            provider={mention.provider}
          />,
          element,
          `${mention.credentialId}-${index}`,
        ),
      )}
    </>
  );
}
