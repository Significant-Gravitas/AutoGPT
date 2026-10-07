"use client";

import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { Button } from "@/components/atoms/Button/Button";
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
        <Button
          key={option.id}
          type="button"
          variant="ghost"
          onClick={() => onPick(option.id)}
          disabled={Boolean(selectedCategory)}
          aria-pressed={selectedCategory ? true : undefined}
          className={cn(
            "flex h-auto min-w-0 px-5 py-2.5 leading-5 text-foreground disabled:text-foreground",
            selectedCategory
              ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
              : "border-border bg-background hover:border-accent hover:bg-accent/5",
          )}
        >
          <span
            aria-hidden
            className="size-3 shrink-0 rounded-full"
            style={{ backgroundColor: option.hex }}
          />
          {option.label}
        </Button>
      ))}
    </div>
  );
}
