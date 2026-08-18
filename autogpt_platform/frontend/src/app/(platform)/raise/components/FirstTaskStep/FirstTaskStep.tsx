"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { bubbleClassFor } from "../ColorStep/helpers";

interface Props {
  name: string;
  color: string | null;
  submittedTask: string | null;
  isSubmitting: boolean;
  onSubmit: (task: string) => void;
  onSkip: () => void;
}

export function FirstTaskStep({
  name,
  color,
  submittedTask,
  isSubmitting,
  onSubmit,
  onSkip,
}: Props) {
  const [value, setValue] = useState("");
  const trimmed = value.trim();

  if (submittedTask !== null) {
    return (
      <p
        className={cn(
          "ml-auto w-fit max-w-[80%] rounded-2xl border px-4 py-3 text-[15px] leading-relaxed text-foreground",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        {submittedTask || "Nothing yet — we'll figure it out together"}
      </p>
    );
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!trimmed) return;
    onSubmit(trimmed);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full flex-col items-end gap-3"
    >
      <Input
        id="raise-first-task"
        type="textarea"
        label="What should they start on?"
        hideLabel
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={`Tell ${name || "them"} what to do first — they'll start on it in your first chat.`}
        rows={3}
        maxLength={2000}
        wrapperClassName="mb-0 w-full max-w-[42rem]"
      />
      <div className="flex items-center gap-2">
        <Button
          type="submit"
          variant="primary"
          size="small"
          disabled={!trimmed || isSubmitting}
          loading={isSubmitting}
          className="h-[2.625rem] rounded-xl py-3"
        >
          {`Bring ${name || "them"} to life`}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="small"
          onClick={onSkip}
          disabled={isSubmitting}
          className="h-[2.625rem] rounded-xl py-3"
        >
          Skip
        </Button>
      </div>
    </form>
  );
}
