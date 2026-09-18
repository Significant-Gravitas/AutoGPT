"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { Forward02Icon } from "@hugeicons/core-free-icons";
import { bubbleClassFor } from "../ColorStep/helpers";
import { JOB_TITLE_MAX_LENGTH, useJobTitleStep } from "./useJobTitleStep";

interface Props {
  selectedTitle: string | null;
  suggestions: string[];
  color: string | null;
  onSubmit: (jobTitle: string) => void;
  onSkip: () => void;
}

export function JobTitleStep({
  selectedTitle,
  suggestions,
  color,
  onSubmit,
  onSkip,
}: Props) {
  const { custom, setCustom, trimmed, submitCustom } = useJobTitleStep({
    onSubmit,
  });

  if (selectedTitle === "") {
    return (
      <div
        className={cn(
          "ml-auto rounded-2xl border px-4 py-3 text-[15px] leading-relaxed text-foreground",
          "flex w-fit items-center gap-2",
          bubbleClassFor(color) ?? "border-accent bg-accent/5",
        )}
      >
        <Icon icon={Forward02Icon} size={16} aria-hidden />
        Skipped
      </div>
    );
  }

  const chips = selectedTitle ? [selectedTitle] : suggestions;

  return (
    <div className="flex flex-col items-end gap-4">
      {chips.length > 0 ? (
        <div
          role="group"
          aria-label="Suggested job titles"
          className="flex flex-wrap justify-end gap-2.5"
        >
          {chips.map((chip) => (
            <button
              key={chip}
              type="button"
              onClick={() => onSubmit(chip)}
              disabled={Boolean(selectedTitle)}
              aria-pressed={selectedTitle ? true : undefined}
              className={cn(
                "rounded-full border px-5 py-2.5 text-sm font-medium text-foreground transition-colors",
                selectedTitle
                  ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
                  : "border-border bg-background hover:border-accent hover:bg-accent/5",
              )}
            >
              {chip}
            </button>
          ))}
        </div>
      ) : null}
      {selectedTitle || chips.length === 0 ? null : (
        <span
          aria-hidden
          className="mr-4 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground"
        >
          or
        </span>
      )}
      {selectedTitle ? null : (
        <form
          onSubmit={submitCustom}
          className="flex flex-wrap items-center justify-end gap-2"
        >
          <Input
            id="raise-job-title"
            label="Job title"
            hideLabel
            size="small"
            value={custom}
            onChange={(event) => setCustom(event.target.value)}
            placeholder="Type a job title…"
            maxLength={JOB_TITLE_MAX_LENGTH}
            wrapperClassName="mb-0 w-full max-w-[16rem] [&_input]:h-[2.625rem] [&_input]:py-3"
          />
          <Button
            type="submit"
            variant="primary"
            size="small"
            disabled={!trimmed}
            className="h-[2.625rem] rounded-xl py-3"
          >
            Add title
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
        </form>
      )}
    </div>
  );
}
