"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { bubbleClassFor } from "../ColorStep/helpers";
import { aboutPlaceholderFor } from "../RoleStep/helpers";

interface Props {
  submittedAbout: string | null;
  role: string | null;
  color: string | null;
  onSubmit: (about: string) => void;
  onSkip: () => void;
}

export function AboutStep({
  submittedAbout,
  role,
  color,
  onSubmit,
  onSkip,
}: Props) {
  const [value, setValue] = useState("");
  const trimmed = value.trim();

  if (submittedAbout !== null) {
    return (
      <p
        className={cn(
          "ml-auto max-w-[80%] rounded-2xl border px-4 py-3 text-[15px] leading-relaxed text-foreground",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        {submittedAbout}
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
        id="raise-about"
        type="textarea"
        label="Anything about your expert"
        hideLabel
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={aboutPlaceholderFor(role)}
        rows={4}
        maxLength={2000}
        wrapperClassName="mb-0 w-full max-w-[42rem]"
      />
      <div className="flex items-center gap-2">
        <Button
          type="submit"
          variant="primary"
          size="small"
          disabled={!trimmed}
          className="h-[2.625rem] rounded-xl py-3"
        >
          {`That's them`}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="small"
          onClick={onSkip}
          className="h-[2.625rem] rounded-xl py-3"
        >
          Skip
        </Button>
      </div>
    </form>
  );
}
