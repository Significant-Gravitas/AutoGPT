"use client";

import { cn } from "@/lib/utils";
import { COLOR_OPTIONS, findColorOption } from "./helpers";

interface Props {
  selectedColor: string | null;
  onPick: (colorId: string) => void;
}

export function ColorStep({ selectedColor, onPick }: Props) {
  const selected = findColorOption(selectedColor);
  const options = selected ? [selected] : COLOR_OPTIONS;

  return (
    <div
      role="group"
      aria-label="Expert color"
      className="flex flex-wrap justify-end gap-2.5"
    >
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          onClick={() => onPick(option.id)}
          disabled={Boolean(selected)}
          aria-label={option.label}
          aria-pressed={selected ? true : undefined}
          className={cn(
            "size-8 rounded-full transition-transform",
            option.swatchClassName,
            selected ? null : "hover:scale-110",
          )}
        />
      ))}
    </div>
  );
}
