"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { isKey } from "@/lib/keyboard";
import { useState, type FormEvent, type KeyboardEvent } from "react";

/** The API's limit on the note. */
const NOTE_MAX = 1000;

interface Props {
  taskID: string;
  isPending: boolean;
  onSubmit: (note: string) => void;
  onCancel: () => void;
}

/** What to change before the task runs again. A note is needed: Enter
 *  sends it, Shift+Enter starts a new line, Escape puts the form away. */
export function RequestChangesForm({
  taskID,
  isPending,
  onSubmit,
  onCancel,
}: Props) {
  const [text, setText] = useState("");
  const [isMissing, setIsMissing] = useState(false);

  function send() {
    const note = text.trim();
    if (!note) {
      setIsMissing(true);
      return;
    }
    onSubmit(note);
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    send();
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) {
    if (isKey(e, "Enter") && !e.shiftKey) {
      e.preventDefault();
      send();
      return;
    }
    if (!isKey(e, "Escape")) return;
    e.preventDefault();
    e.stopPropagation();
    onCancel();
  }

  return (
    <form
      aria-label="Request changes"
      onSubmit={submit}
      className="flex flex-col gap-2"
    >
      <Input
        id={`request-changes-${taskID}`}
        type="textarea"
        label="What should change?"
        size="small"
        rows={2}
        value={text}
        maxLength={NOTE_MAX}
        error={isMissing ? "Say what to change." : undefined}
        onChange={(e) => {
          setText(e.target.value);
          setIsMissing(false);
        }}
        onKeyDown={onKeyDown}
        autoFocus
      />
      <span className="flex justify-end gap-2">
        <Button
          type="button"
          variant="ghost-muted"
          size="xs"
          onClick={onCancel}
        >
          Cancel
        </Button>
        <Button type="submit" variant="primary" size="xs" loading={isPending}>
          Send
        </Button>
      </span>
    </form>
  );
}
