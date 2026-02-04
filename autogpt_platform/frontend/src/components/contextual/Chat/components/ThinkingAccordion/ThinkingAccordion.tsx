"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { TypeWriter } from "@/components/molecules/Typewriter/Typewriter";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import { ThinkingAccordionAnimation } from "./components/ThinkingAccordionAnimation";

const THINKING_LABELS = [
  "Hmm, let me think…",
  "Cooking up something good…",
  "Connecting the dots…",
  "Brewing ideas…",
  "Down the rabbit hole…",
  "Following the thread…",
  "Almost there, maybe…",
  "One sec, got an idea…",
  "Piecing it together…",
  "Digging deeper…",
  "Let me check that…",
  "Working on it…",
  "Hold that thought…",
  "Getting there…",
] as const;

const LABEL_ROTATION_INTERVAL = 4000;

export interface Props {
  chunks: string[];
  className?: string;
}

/**
 * ThinkingIndicator displays a "Thinking..." indicator during streaming.
 * Clicking opens a Dialog to view reasoning chunks.
 *
 * ChatGPT-style UX:
 * - During streaming: Shows indicator with rotating status label
 * - User can click to open a dialog and see live-updating reasoning
 * - Tool responses are shown separately via clickable tool_call messages
 * - After streaming ends: This component should be unmounted
 */
export function ThinkingAccordion({ chunks, className }: Props) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [currentLabel, setCurrentLabel] = useState<string>(
    () => THINKING_LABELS[Math.floor(Math.random() * THINKING_LABELS.length)],
  );

  // Rotate labels randomly
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentLabel((prev) => {
        let next: string;
        do {
          next =
            THINKING_LABELS[Math.floor(Math.random() * THINKING_LABELS.length)];
        } while (next === prev);
        return next;
      });
    }, LABEL_ROTATION_INTERVAL);

    return () => clearInterval(interval);
  }, []);
  const displayText = chunks.join("");

  return (
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3 px-4 py-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Clickable thinking indicator */}
          <button
            type="button"
            onClick={() => setIsDialogOpen(true)}
            className="inline-flex w-fit cursor-pointer items-center gap-2 rounded-md px-1 py-2 transition-colors hover:bg-neutral-100"
          >
            <ThinkingAccordionAnimation />
            <TypeWriter
              text={currentLabel}
              className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-sm font-medium text-transparent"
            />
          </button>

          {/* Dialog for viewing reasoning content */}
          <Dialog
            title={
              <span className="inline-flex items-center gap-2">
                <ThinkingAccordionAnimation />
                <span className="text-sm font-medium text-neutral-600">
                  Reasoning
                </span>
              </span>
            }
            controlled={{
              isOpen: isDialogOpen,
              set: setIsDialogOpen,
            }}
            onClose={() => setIsDialogOpen(false)}
            styling={{ maxWidth: 600, width: "100%", minWidth: "auto" }}
          >
            <Dialog.Content>
              <div className="max-h-[60vh] overflow-y-auto text-left text-[1rem] leading-relaxed">
                {displayText ? (
                  <MarkdownContent content={displayText} />
                ) : (
                  <span className="text-neutral-400">
                    Gathering thoughts...
                  </span>
                )}
              </div>
            </Dialog.Content>
          </Dialog>
        </div>
      </div>
    </div>
  );
}
