import { Input } from "@/components/atoms/Input/Input";
import { LongNumberHint } from "@/components/contextual/LongNumberHint/LongNumberHint";
import { hasLongWholeNumber, isJsonTextFor } from "@/lib/json-text";
import { useId, useState } from "react";

interface Props {
  /** Compared by reference to spot an outside change, so keep it stable
   *  while it is unchanged; an equal copy on every render would replace a
   *  draft that doesn't parse yet. */
  value: unknown;
  /** Text to start from instead of `value`: a draft kept from before the
   *  card remounted, which may not parse. */
  initialText?: string;
  onChange: (value: unknown) => void;
  /** The draft no longer parses. Its text goes up as is so approval can be
   *  blocked: the last valid value must not be sent while the screen shows
   *  something else. */
  onInvalidDraft: (text: string) => void;
}

/** The payload as editable JSON. The text is the user's draft: an edit that
 *  doesn't parse yet (every half-typed one) stays on screen and is reported
 *  once it does. Rebuilding the text from the value would drop those edits,
 *  and rewriting it under the user throws the caret to the end and cancels
 *  an IME composition. */
export function JsonDataInput({
  value,
  initialText,
  onChange,
  onInvalidDraft,
}: Props) {
  const id = useId();
  const [text, setText] = useState(() => initialText ?? toJsonText(value));
  const [shownValue, setShownValue] = useState(value);

  if (value !== shownValue) {
    setShownValue(value);
    if (!text.trim() || !isJsonTextFor(text, value)) setText(toJsonText(value));
  }

  function handleChange(next: string) {
    setText(next);
    try {
      onChange(JSON.parse(next));
    } catch {
      onInvalidDraft(next);
    }
  }

  const longNumberHintId = hasLongWholeNumber(text)
    ? `${id}-long-number`
    : undefined;

  return (
    <div className="mb-6 flex flex-col gap-2">
      <Input
        id={id}
        label="Value"
        hideLabel
        size="small"
        type="textarea"
        rows={6}
        value={text}
        onChange={(e) => handleChange(e.target.value)}
        placeholder="Edit JSON data"
        aria-describedby={longNumberHintId}
        wrapperClassName="mb-0"
        className="font-mono text-body"
      />
      <LongNumberHint text={text} id={longNumberHintId} />
    </div>
  );
}

function toJsonText(value: unknown) {
  return JSON.stringify(value, null, 2);
}
