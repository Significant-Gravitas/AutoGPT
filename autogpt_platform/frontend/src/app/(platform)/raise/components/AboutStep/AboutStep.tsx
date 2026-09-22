"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { Forward02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { bubbleClassFor } from "../ColorStep/helpers";
import { aboutPlaceholderFor } from "../RoleStep/helpers";

interface Props {
  submittedAbout: string | null;
  name: string | null;
  color: string | null;
  onSubmit: (about: string) => void;
  onSkip: () => void;
}

export function AboutStep({
  submittedAbout,
  name,
  color,
  onSubmit,
  onSkip,
}: Props) {
  const [value, setValue] = useState("");
  const trimmed = value.trim();

  if (submittedAbout !== null) {
    const isSkipped = submittedAbout === "";

    return (
      <div
        className={cn(
          "ml-auto max-w-[80%] rounded-2xl border px-4 py-3 text-[15px] leading-relaxed text-foreground",
          isSkipped && "flex w-fit items-center gap-2",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        {isSkipped ? (
          <>
            <Icon icon={Forward02Icon} size={16} aria-hidden />
            Skipped
          </>
        ) : (
          submittedAbout
        )}
      </div>
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
        placeholder={aboutPlaceholderFor(name)}
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
          That&apos;s it
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
