import { Input } from "@/components/atoms/Input/Input";
import { isJsonTextFor } from "@/lib/json-text";
import { useState } from "react";

interface Props {
  value: unknown;
  onChange: (value: unknown) => void;
}

/** The payload as editable JSON. The text is the user's draft: an edit that
 *  doesn't parse yet (every half-typed one) stays on screen and is reported
 *  once it does. Rebuilding the text from the value would drop those edits,
 *  and rewriting it under the user throws the caret to the end and cancels
 *  an IME composition. */
export function JsonDataInput({ value, onChange }: Props) {
  const [text, setText] = useState(() => toJsonText(value));
  const [shownValue, setShownValue] = useState(value);

  if (value !== shownValue) {
    setShownValue(value);
    if (!isJsonTextFor(text, value)) setText(toJsonText(value));
  }

  function handleChange(next: string) {
    setText(next);
    try {
      onChange(JSON.parse(next));
    } catch {}
  }

  return (
    <Input
      id="data-json"
      label="Value"
      hideLabel
      size="small"
      type="textarea"
      rows={6}
      value={text}
      onChange={(e) => handleChange(e.target.value)}
      placeholder="Edit JSON data"
      className="font-mono text-sm"
    />
  );
}

function toJsonText(value: unknown) {
  return JSON.stringify(value, null, 2);
}
