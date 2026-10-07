import { cn } from "@/lib/utils";
import React from "react";

interface Props {
  children: React.ReactNode;
  className?: string;
}

export function CardWrapper({ children, className }: Props) {
  return (
    <div className={cn("relative my-6 overflow-hidden rounded-xl", className)}>
      {/* 1px gradient border: the card sits on a gradient-filled layer */}
      <div
        className="absolute inset-0 rounded-xl p-px"
        style={{
          background:
            "linear-gradient(135deg, var(--color-zinc-100), var(--color-zinc-200), var(--color-zinc-100))",
        }}
      >
        <div className="h-full w-full rounded-xl bg-card" />
      </div>
      {children}
    </div>
  );
}
