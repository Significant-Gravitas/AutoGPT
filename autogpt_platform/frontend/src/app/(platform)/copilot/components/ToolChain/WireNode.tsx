"use client";

import { cn } from "@/lib/utils";

interface Props {
  isLast: boolean;
  testId: string;
  tone?: "default" | "waiting";
  busy?: boolean;
  label?: string;
  children: React.ReactNode;
}

/** A card that sits ON the chain's wire instead of under a row: the wire
 *  runs into its top edge and, unless it ends the chain, out of its bottom. */
export function WireNode({
  isLast,
  testId,
  tone = "default",
  busy,
  label,
  children,
}: Props) {
  return (
    <div className="flex flex-col" data-testid={testId}>
      <span aria-hidden className="ml-[14px] h-3 w-px bg-zinc-200" />
      <div
        aria-busy={busy}
        aria-label={label}
        className={cn(
          "w-full overflow-hidden rounded-xl border bg-white",
          tone === "waiting" ? "border-amber-500/35" : "border-zinc-200",
        )}
      >
        {children}
      </div>
      {!isLast && (
        <span aria-hidden className="ml-[14px] h-3 w-px bg-zinc-200" />
      )}
    </div>
  );
}
