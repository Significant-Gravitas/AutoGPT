"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { bubbleClassFor } from "../ColorStep/helpers";
import { useNameStep } from "./useNameStep";

interface Props {
  selectedName: string | null;
  suggestions: string[];
  color: string | null;
  onSubmit: (name: string) => void;
}

export function NameStep({
  selectedName,
  suggestions,
  color,
  onSubmit,
}: Props) {
  const { custom, setCustom, trimmed, submitCustom } = useNameStep({
    onSubmit,
  });
  // Once a name is picked the step collapses to that one chip, held in the
  // state it had on hover.
  const chips = selectedName ? [selectedName] : suggestions;

  return (
    <div className="flex flex-col items-end gap-4">
      <div
        role="group"
        aria-label="Suggested names"
        className="flex flex-wrap justify-end gap-2.5"
      >
        {chips.map((chip) => (
          <button
            key={chip}
            type="button"
            onClick={() => onSubmit(chip)}
            disabled={Boolean(selectedName)}
            aria-pressed={selectedName ? true : undefined}
            className={cn(
              "rounded-full border px-5 py-2.5 text-sm font-medium text-foreground transition-colors",
              selectedName
                ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
                : "border-border bg-background hover:border-accent hover:bg-accent/5",
            )}
          >
            {chip}
          </button>
        ))}
      </div>
      {selectedName ? null : (
        <span
          aria-hidden
          className="mr-4 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground"
        >
          or
        </span>
      )}
      {selectedName ? null : (
        <form onSubmit={submitCustom} className="flex items-center gap-2">
          <Input
            id="raise-name"
            label="Or type your own"
            hideLabel
            size="small"
            value={custom}
            onChange={(event) => setCustom(event.target.value)}
            placeholder="Type a name…"
            maxLength={100}
            wrapperClassName="mb-0 w-full max-w-[16rem] [&_input]:h-[2.625rem] [&_input]:py-3"
          />
          <Button
            type="submit"
            variant="primary"
            size="small"
            disabled={!trimmed}
            className="h-[2.625rem] rounded-xl py-3"
          >
            Name me
          </Button>
        </form>
      )}
    </div>
  );
}
