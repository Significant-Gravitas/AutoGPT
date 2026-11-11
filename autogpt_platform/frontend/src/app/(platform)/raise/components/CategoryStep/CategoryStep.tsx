"use client";

import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { cn } from "@/lib/utils";
import { bubbleClassFor } from "../ColorStep/helpers";
import { categoryOptionsForSelection } from "./helpers";

interface Props {
  selectedCategory: ExpertAvatarRequestCategory | null;
  color: string | null;
  onPick: (category: ExpertAvatarRequestCategory) => void;
}

export function CategoryStep({ selectedCategory, color, onPick }: Props) {
  const options = categoryOptionsForSelection(selectedCategory);

  return (
    <div
      role="group"
      aria-label="What the expert works on"
      className="flex flex-wrap justify-end gap-2.5"
    >
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          onClick={() => onPick(option.id)}
          disabled={Boolean(selectedCategory)}
          aria-pressed={selectedCategory ? true : undefined}
          className={cn(
            "flex items-center gap-2 rounded-full border px-5 py-2.5 text-body font-medium text-foreground-base transition-colors",
            selectedCategory
              ? (bubbleClassFor(color) ?? "border-brand bg-brand/5")
              : "border-border-base bg-canvas hover:border-brand hover:bg-brand/5",
          )}
        >
          <span
            aria-hidden
            className="size-3 shrink-0 rounded-full"
            // oxlint-disable-next-line design-system/no-inline-styles -- the swatch is the category's own colour, from data
            style={{ backgroundColor: option.hex }}
          />
          {option.label}
        </button>
      ))}
    </div>
  );
}
