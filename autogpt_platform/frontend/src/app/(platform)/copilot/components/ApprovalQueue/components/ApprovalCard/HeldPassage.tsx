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
          "border-l-2 border-warning bg-warning-surface/60 px-3 py-2 text-sm [overflow-wrap:anywhere] whitespace-pre-wrap text-foreground-emphasis",
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
          // oxlint-disable-next-line design-system/no-restyle -- the link variant takes the default size's padding; the toggle sits flush under the quote
          className="h-auto min-w-0 self-start px-0 py-0"
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
