"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { FormEvent } from "react";
import { FIELD } from "./styles";

interface Props {
  value: string;
  onChange(value: string): void;
  onSubmit(): void;
  onCancel(): void;
  isActing: boolean;
}

export function GoalForm({
  value,
  onChange,
  onSubmit,
  onCancel,
  isActing,
}: Props) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2">
      <input
        type="text"
        autoFocus
        aria-label="New goal"
        className={FIELD}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={isActing}
      />
      <div className="flex gap-1.5">
        <Button
          type="submit"
          variant="primary"
          size="xs"
          loading={isActing}
          disabled={!value.trim()}
        >
          Change goal
        </Button>
        <Button
          type="button"
          variant="ghost-muted"
          size="xs"
          onClick={onCancel}
        >
          Cancel
        </Button>
      </div>
    </form>
  );
}
