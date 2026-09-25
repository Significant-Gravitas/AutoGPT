"use client";

import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { cn } from "@/lib/utils";
import { bubbleClassFor } from "../ColorStep/helpers";
import { categoryOptionsForSelection } from "./helpers";

interface Props {
  selectedCategory: ExpertAvatarRequestCategory | null;
  suggested: ExpertAvatarRequestCategory;
  color: string | null;
  onPick: (category: ExpertAvatarRequestCategory) => void;
}

export function CategoryStep({
  selectedCategory,
  suggested,
  color,
  onPick,
}: Props) {
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
            "flex items-center gap-2 rounded-full border px-5 py-2.5 text-sm font-medium text-foreground transition-colors",
            selectedCategory
              ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
              : "border-border bg-background hover:border-accent hover:bg-accent/5",
            !selectedCategory && option.id === suggested && "border-accent",
          )}
        >
          <span
            aria-hidden
            className="size-3 shrink-0 rounded-full"
            style={{ backgroundColor: option.hex }}
          />
          {option.label}
        </button>
      ))}
    </div>
  );
}
