"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { cn } from "@/lib/utils";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { useRef } from "react";
import { useTextReveal } from "../../useTextReveal";

interface Props {
  prompt: string | null;
  isStreaming: boolean;
  onSend: () => void;
}

export function TourPromptBar({ prompt, isStreaming, onSend }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  // The parent re-keys this component per turn, so the reveal restarts
  // whenever a new prompt prefills — it types itself in like the transcript.
  const { visibleText, isDone: isTyped } = useTextReveal(prompt ?? "");
  // No prompt means the turn plays itself (auto-started first turn) or the
  // demo is over — the bar sits empty and disabled either way.
  const isDisabled = isStreaming || !prompt;

  // The parent re-keys this component whenever the prefilled prompt changes, so
  // focusing on mount keeps the box ready for the visitor to just press Enter.
  useMountEffect(() => {
    if (!isDisabled) ref.current?.focus();
  });

  function send() {
    if (!isDisabled) onSend();
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      send();
    }
  }

  // The bar isn't a text field — it shows a fixed prompt and only sends on
  // click/Enter — so it's exposed as a button with a "Send: …" label.
  return (
    <div
      ref={ref}
      role="button"
      tabIndex={isDisabled ? -1 : 0}
      aria-label={prompt ? `Send: ${prompt}` : "Autopilot is working"}
      aria-disabled={isDisabled}
      onKeyDown={handleKeyDown}
      onClick={send}
      className={cn(
        "flex cursor-pointer items-center gap-3 rounded-xl border border-zinc-200 bg-white px-5 py-3.5 shadow-[0_2px_8px_rgba(0,0,0,0.04),0_0_32px_-4px_rgba(99,102,241,0.4)] outline-none transition-shadow focus-visible:border-zinc-300 focus-visible:shadow-[0_2px_8px_rgba(0,0,0,0.04),0_0_44px_-2px_rgba(99,102,241,0.55)]",
        isDisabled && "cursor-default opacity-60",
      )}
    >
      <span className="min-h-6 flex-1 truncate text-base text-zinc-700">
        {visibleText}
        {prompt && !isTyped && (
          <span className="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-zinc-300 align-middle" />
        )}
      </span>
      {!isDisabled && isTyped && (
        <span className="hidden shrink-0 items-center gap-1 text-xs text-zinc-400 sm:flex">
          Press
          <kbd className="rounded border border-zinc-200 bg-zinc-50 px-1.5 py-0.5 font-sans text-[0.7rem] text-zinc-500">
            Enter
          </kbd>
        </span>
      )}
      <span
        aria-hidden="true"
        className={cn(
          "flex size-10 shrink-0 items-center justify-center rounded-full text-white transition-colors",
          isDisabled ? "bg-zinc-200" : "bg-zinc-800",
        )}
      >
        <ArrowUpIcon className="size-4" weight="bold" />
      </span>
    </div>
  );
}
