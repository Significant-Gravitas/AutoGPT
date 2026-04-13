import { PaperPlaneTilt, StopCircle } from "@phosphor-icons/react";
import { KeyboardEvent } from "react";

/** Max characters permitted in the chat textarea. Enforced both in the UI and by the backend. */
export const TEXTAREA_MAX_LENGTH = 4000;
/** Show the character counter once the user reaches this fraction of the max. */
const CHAR_COUNT_WARNING_RATIO = 0.8;

interface Props {
  value: string;
  onChange: (v: string) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  isDisabled: boolean;
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
}

export function PanelInput({
  value,
  onChange,
  onKeyDown,
  onSend,
  onStop,
  isStreaming,
  isDisabled,
  textareaRef,
}: Props) {
  const charCount = value.length;
  const showCharCount =
    charCount >= TEXTAREA_MAX_LENGTH * CHAR_COUNT_WARNING_RATIO;
  const atLimit = charCount >= TEXTAREA_MAX_LENGTH;

  return (
    <div className="border-t border-slate-100 p-3">
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          disabled={isDisabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your agent... (Enter to send, Shift+Enter for newline)"
          rows={2}
          maxLength={TEXTAREA_MAX_LENGTH}
          aria-label="Chat message"
          className="flex-1 resize-none rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-200 disabled:opacity-50"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-red-100 text-red-600 transition-colors hover:bg-red-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2"
            aria-label="Stop"
          >
            <StopCircle size={18} />
          </button>
        ) : (
          <button
            type="button"
            onClick={onSend}
            disabled={isDisabled || !value.trim()}
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-violet-600 text-white transition-colors hover:bg-violet-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2 disabled:opacity-40"
            aria-label="Send"
          >
            <PaperPlaneTilt size={18} />
          </button>
        )}
      </div>
      {showCharCount && (
        <div
          className={
            "mt-1 text-right text-[11px] " +
            (atLimit ? "text-red-600" : "text-slate-400")
          }
          aria-live="polite"
        >
          {charCount} / {TEXTAREA_MAX_LENGTH}
        </div>
      )}
    </div>
  );
}
