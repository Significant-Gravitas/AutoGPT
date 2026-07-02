"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { cn } from "@/lib/utils";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { useRef } from "react";
import { TourUpsellCard } from "../TourUpsellCard/TourUpsellCard";

interface Props {
  prompt: string | null;
  isStreaming: boolean;
  isExhausted: boolean;
  onSend: () => void;
  onReplay: () => void;
}

export function TourPromptBar({
  prompt,
  isStreaming,
  isExhausted,
  onSend,
  onReplay,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  // The parent re-keys this component whenever the prefilled prompt changes, so
  // focusing on mount keeps the box ready for the visitor to just press Enter.
  useMountEffect(() => {
    if (!isStreaming && !isExhausted) ref.current?.focus();
  });

  if (isExhausted) {
    return <TourUpsellCard onReplay={onReplay} />;
  }

  function send() {
    if (!isStreaming) onSend();
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
      tabIndex={0}
      aria-label={prompt ? `Send: ${prompt}` : "Send message"}
      aria-disabled={isStreaming}
      onKeyDown={handleKeyDown}
      onClick={send}
      className={cn(
        "flex cursor-pointer items-center gap-3 rounded-xl border border-zinc-200 bg-white px-5 py-3.5 shadow-[0_2px_8px_rgba(0,0,0,0.04),0_0_32px_-4px_rgba(99,102,241,0.4)] outline-none transition-shadow focus-visible:border-zinc-300 focus-visible:shadow-[0_2px_8px_rgba(0,0,0,0.04),0_0_44px_-2px_rgba(99,102,241,0.55)]",
        isStreaming && "cursor-default opacity-60",
      )}
    >
      <span className="flex-1 truncate text-base text-zinc-700">{prompt}</span>
      <span className="hidden shrink-0 items-center gap-1 text-xs text-zinc-400 sm:flex">
        Press
        <kbd className="rounded border border-zinc-200 bg-zinc-50 px-1.5 py-0.5 font-sans text-[0.7rem] text-zinc-500">
          Enter
        </kbd>
      </span>
      <span
        aria-hidden="true"
        className={cn(
          "flex size-10 shrink-0 items-center justify-center rounded-full text-white transition-colors",
          isStreaming ? "bg-zinc-200" : "bg-zinc-800",
        )}
      >
        <ArrowUpIcon className="size-4" weight="bold" />
      </span>
    </div>
  );
}
