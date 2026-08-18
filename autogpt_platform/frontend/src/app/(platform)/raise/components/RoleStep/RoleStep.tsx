"use client";

import { cn } from "@/lib/utils";
import { bubbleClassFor } from "../ColorStep/helpers";
import { findRoleOption, ROLE_OPTIONS } from "./helpers";

interface Props {
  selectedRole: string | null;
  color: string | null;
  onPick: (roleId: string) => void;
}

export function RoleStep({ selectedRole, color, onPick }: Props) {
  const selected = findRoleOption(selectedRole);
  const options = selected ? [selected] : ROLE_OPTIONS;

  return (
    <div
      role="group"
      aria-label="What the expert does"
      className="flex flex-wrap justify-end gap-2.5"
    >
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          onClick={() => onPick(option.id)}
          disabled={Boolean(selected)}
          aria-pressed={selected ? true : undefined}
          className={cn(
            "rounded-full border px-5 py-2.5 text-sm font-medium text-foreground transition-colors",
            selected
              ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
              : "border-border bg-background hover:border-accent hover:bg-accent/5",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
