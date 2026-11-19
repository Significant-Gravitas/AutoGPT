"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { isKey } from "@/lib/keyboard";
import { useState, type FormEvent, type KeyboardEvent } from "react";

/** The API's limit on what the user asks to change. */
const INSTRUCTIONS_MAX = 2000;

interface Props {
  taskID: string;
  isPending: boolean;
  onSubmit: (instructions: string) => void;
  onCancel: () => void;
}

/** Before a task runs again: an optional note on what to do differently. */
export function TryAgainForm({ taskID, isPending, onSubmit, onCancel }: Props) {
  const [text, setText] = useState("");

  function submit(e: FormEvent) {
    e.preventDefault();
    onSubmit(text);
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) {
    if (!isKey(e, "Escape")) return;
    e.preventDefault();
    onCancel();
  }

  return (
    <form
      aria-label="Try the task again"
      onSubmit={submit}
      className="flex flex-col gap-2"
    >
      <Input
        id={`try-again-${taskID}`}
        type="textarea"
        label="Anything to do differently?"
        placeholder="Optional"
        size="small"
        rows={2}
        value={text}
        maxLength={INSTRUCTIONS_MAX}
        onChange={(e) => setText(e.target.value)}
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
          Try again
        </Button>
      </span>
    </form>
  );
}
