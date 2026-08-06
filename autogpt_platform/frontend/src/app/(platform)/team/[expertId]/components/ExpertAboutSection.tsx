"use client";

import { cn } from "@/lib/utils";
import { useState } from "react";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

// Roughly the number of characters that fit in the four-line clamp below.
const CLAMPED_BIO_LENGTH = 280;

interface Props {
  text: string;
}

export function ExpertAboutSection({ text }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isClampable = text.length > CLAMPED_BIO_LENGTH;

  return (
    <section>
      <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
        About
      </div>
      <p
        className={cn(
          "max-w-prose whitespace-pre-line text-base leading-relaxed text-zinc-600",
          isClampable && !isExpanded && "line-clamp-4",
        )}
      >
        {text}
      </p>
      {isClampable ? (
        <button
          type="button"
          onClick={() => setIsExpanded((v) => !v)}
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
