"use client";
import { cn } from "@/lib/utils";
import React, { ReactNode } from "react";

interface Props extends React.HTMLProps<HTMLDivElement> {
  children?: ReactNode;
  showRadialGradient?: boolean;
}

export function AuroraBackground({
  className,
  children,
  showRadialGradient = true,
  ...props
}: Props) {
  return (
    <div
      className={cn(
        "transition-bg relative flex flex-col items-center justify-center overflow-hidden",
        className,
      )}
      {...props}
    >
      <div
        aria-hidden
        className="absolute inset-0 overflow-hidden"
        style={
          {
            "--aurora":
              "repeating-linear-gradient(100deg, #3b82f6 10%, #a5b4fc 15%, #93c5fd 20%, #ddd6fe 25%, #60a5fa 30%)",
            "--white-gradient":
              "repeating-linear-gradient(100deg, #fff 0%, #fff 7%, transparent 10%, transparent 12%, #fff 16%)",
          } as React.CSSProperties
        }
      >
        <div
          className={cn(
            `pointer-events-none absolute inset-[-10px] [background-image:var(--white-gradient),var(--aurora)] bg-size-[300%,200%] bg-position-[50%_50%,50%_50%] opacity-10 blur-[10px] invert filter will-change-transform after:absolute after:inset-0 after:animate-aurora after:[background-image:var(--white-gradient),var(--aurora)] after:bg-size-[200%,100%] after:bg-fixed after:mix-blend-difference after:content-[""]`,
            showRadialGradient &&
              `mask-[radial-gradient(ellipse_at_100%_0%,black_10%,transparent_70%)]`,
          )}
        />
      </div>
      {children}
    </div>
  );
}
