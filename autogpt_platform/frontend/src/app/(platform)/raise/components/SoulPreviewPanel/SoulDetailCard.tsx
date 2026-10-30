"use client";

import { cn } from "@/lib/utils";
import { textClassFor } from "../ColorStep/helpers";

type Props = {
  label: string;
  value: string;
  color: string | null;
};

// Each answer arrives as its own card sliding up under the identity card, so
// the stack grows downward and lifts the name as it does.
export function SoulDetailCard({ label, value, color }: Props) {
  return (
    <div className="w-full rounded-3xl border border-border bg-background px-6 py-4 shadow-lg duration-500 animate-in fade-in fill-mode-both slide-in-from-bottom-6 motion-reduce:animate-none">
      <p
        className={cn(
          "mb-1 text-xs font-semibold tracking-caps uppercase",
          textClassFor(color) ?? "text-muted-foreground",
        )}
      >
        {label}
      </p>
      <p className="line-clamp-3 text-body-md leading-relaxed text-foreground">
        {value}
      </p>
    </div>
  );
}
