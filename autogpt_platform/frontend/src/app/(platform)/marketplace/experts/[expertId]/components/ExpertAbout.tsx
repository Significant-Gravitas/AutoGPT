"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { ExpertSectionLabel } from "./ExpertSectionLabel";

// Roughly the number of characters that fit in the six-line clamp below.
const CLAMPED_LENGTH = 420;

interface Props {
  text: string;
}

export function ExpertAbout({ text }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isClampable = text.length > CLAMPED_LENGTH;

  return (
    <section>
      <ExpertSectionLabel>About</ExpertSectionLabel>
      <p
        className={cn(
          "whitespace-pre-line text-base leading-relaxed text-zinc-600",
          isClampable && !isExpanded && "line-clamp-6",
        )}
      >
        {text}
      </p>
      {isClampable ? (
        <button
          type="button"
          onClick={() => setIsExpanded((value) => !value)}
          className="mt-2 flex items-center gap-1 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
        >
          {isExpanded ? "Show less" : "Read more"}
          <Icon
            icon={ArrowDown01Icon}
            size={14}
            className={cn(
              "transition-transform duration-200",
              isExpanded && "rotate-180",
            )}
          />
        </button>
      ) : null}
    </section>
  );
}
