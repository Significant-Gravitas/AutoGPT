"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { useTypewriter } from "./useTypewriter";

type Props = {
  id?: string;
  text: string;
  animate?: boolean;
  onTypingComplete?: () => void;
};

export function AutoGPTBubble({
  id,
  text,
  animate = true,
  onTypingComplete,
}: Props) {
  const { typed, isTyping } = useTypewriter(text, onTypingComplete, animate);

  return (
    <div
      id={id}
      className="flex max-w-[85%] flex-col gap-1.5 self-start duration-500 animate-in fade-in slide-in-from-bottom-3 fill-mode-both motion-reduce:animate-none"
    >
      <div className="flex items-end gap-2">
        <AutoGPTLogo
          hideText
          viewBox="47 -1 42 42"
          className="size-5 shrink-0"
        />
        <span className="text-sm font-medium text-foreground">Autopilot</span>
      </div>
      <p className="text-[15px] leading-relaxed text-foreground">
        <span aria-hidden>{typed}</span>
        {isTyping ? (
          <span
            aria-hidden
            className="ml-0.5 inline-block h-[1.05em] w-px translate-y-[0.15em] animate-pulse bg-foreground/70"
          />
        ) : null}
        {/* The visible text arrives a character at a time; the live region
            should announce the whole line once instead of stuttering. */}
        <span className="sr-only">{text}</span>
      </p>
    </div>
  );
}
