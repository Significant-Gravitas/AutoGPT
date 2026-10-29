"use client";

import { useId, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { passageLines } from "../../helpers";

interface Props {
  passage: string;
}

export function HeldPassage({ passage }: Props) {
  const [expanded, setExpanded] = useState(false);
  const id = useId();
  const lines = passageLines(passage);
  const long = isLong(lines.map((line) => line.text));

  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-xs text-foreground-tertiary">
        What it says
      </figcaption>
      <blockquote
        id={id}
        className={cn(
          "whitespace-pre-wrap border-l-2 border-warning bg-warning-surface/60 px-3 py-2 text-sm text-foreground-emphasis [overflow-wrap:anywhere]",
          long &&
            !expanded &&
            "max-h-40 overflow-hidden [mask-image:linear-gradient(to_bottom,black_70%,transparent)]",
        )}
      >
        {lines.map((line, index) =>
          line.heading ? (
            <strong key={index} className="block font-semibold">
              {line.text}
            </strong>
          ) : (
            <span key={index} className="block">
              {line.text}
            </span>
          ),
        )}
      </blockquote>
      {long && (
        <Button
          type="button"
          variant="link"
          size="small"
          className="h-auto min-w-0 self-start"
          aria-expanded={expanded}
          aria-controls={id}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Show less" : "Show more"}
        </Button>
      )}
    </figure>
  );
}

// About what fits the collapsed height; a quoted sentence never reaches it.
function isLong(lines: string[]) {
  return lines.length > 7 || lines.join("").length > 560;
}
