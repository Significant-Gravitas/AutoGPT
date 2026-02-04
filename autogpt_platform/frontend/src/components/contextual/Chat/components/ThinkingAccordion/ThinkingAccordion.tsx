"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";
import { useEffect, useState } from "react";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";

/**
 * Rotating status labels shown in the accordion trigger during streaming.
 */
const THINKING_LABELS = [
  "Thinking…",
  "Reasoning…",
  "Analyzing…",
  "Working it out…",
  "Cooking…",
  "Connecting the dots…",
  "Mulling it over…",
  "Brewing some ideas…",
  "Putting pieces together…",
  "Down the rabbit hole…",
  "Crunching possibilities…",
  "Following the thread…",
  "Almost there, maybe…",
] as const;

/** Rotation interval for status labels in milliseconds */
const LABEL_ROTATION_INTERVAL = 4000;

/** Typing speed in milliseconds per character */
const TYPING_SPEED = 50;

/**
 * TypeWriter component - reveals text character by character
 */
function TypeWriter({ text, className }: { text: string; className?: string }) {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    setDisplayedText("");
    let currentIndex = 0;

    const interval = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayedText(text.slice(0, currentIndex + 1));
        currentIndex++;
      } else {
        clearInterval(interval);
      }
    }, TYPING_SPEED);

    return () => clearInterval(interval);
  }, [text]);

  return (
    <span className={className}>
      {displayedText}
      <span className="animate-pulse">|</span>
    </span>
  );
}

/**
 * Pulsing loader animation - shows concentric circles pulsing in/out.
 * Size is relative to the text (double the text-sm size = ~1.75rem).
 */
function PulseLoader() {
  return (
    <span
      className="relative flex shrink-0 items-center justify-center"
      style={{ width: "1.75rem", height: "1.75rem" }}
    >
      {/* Inner pulse - shrinks inward */}
      <span
        className="absolute rounded-full"
        style={{
          width: "100%",
          height: "100%",
          boxShadow: "inset 0 0 0 0.25rem #737373",
          animation: "pulseIn 1.8s ease-in-out infinite",
        }}
      />
      {/* Outer pulse - expands outward */}
      <span
        className="absolute rounded-full"
        style={{
          width: "calc(100% - 0.5rem)",
          height: "calc(100% - 0.5rem)",
          boxShadow: "0 0 0 0 #737373",
          animation: "pulseOut 1.8s ease-in-out infinite",
        }}
      />
      {/* Keyframes injected via style tag */}
      <style jsx>{`
        @keyframes pulseIn {
          0% {
            box-shadow: inset 0 0 0 0.25rem #737373;
            opacity: 1;
          }
          50%,
          100% {
            box-shadow: inset 0 0 0 0 #737373;
            opacity: 0;
          }
        }
        @keyframes pulseOut {
          0%,
          50% {
            box-shadow: 0 0 0 0 #737373;
            opacity: 0;
          }
          100% {
            box-shadow: 0 0 0 0.25rem #737373;
            opacity: 1;
          }
        }
      `}</style>
    </span>
  );
}

export interface ThinkingAccordionProps {
  /** Array of reasoning/thinking chunks to display inside the accordion */
  chunks: string[];
  /** Optional className for the container */
  className?: string;
}

/**
 * ThinkingAccordion displays a collapsible "Thinking..." accordion during streaming.
 *
 * ChatGPT-style UX:
 * - During streaming: Shows collapsed by default with rotating status label
 * - User can click to expand and see live-updating reasoning chunks
 * - After streaming ends: This component should be unmounted (reasoning hidden)
 */
export function ThinkingAccordion({
  chunks,
  className,
}: ThinkingAccordionProps) {
  const [labelIndex, setLabelIndex] = useState(() =>
    Math.floor(Math.random() * THINKING_LABELS.length),
  );

  useEffect(() => {
    const interval = setInterval(() => {
      setLabelIndex((prev) => {
        let next: number;
        do {
          next = Math.floor(Math.random() * THINKING_LABELS.length);
        } while (next === prev);
        return next;
      });
    }, LABEL_ROTATION_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  const currentLabel = THINKING_LABELS[labelIndex];
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
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="thinking" className="group/accordion border-none">
              {/* Hide the built-in chevron with [&>svg]:hidden, use our own with proper rotation */}
              <AccordionTrigger className="gap-2 py-2 hover:no-underline [&>svg]:hidden">
                <span className="inline-flex items-center gap-2">
                  <PulseLoader />
                  <TypeWriter
                    text={currentLabel}
                    className="inline-block animate-shimmer bg-gradient-to-r from-neutral-400 via-neutral-600 to-neutral-400 bg-[length:200%_100%] bg-clip-text text-sm font-medium text-transparent"
                  />
                  <ChevronDown className="h-4 w-4 shrink-0 text-neutral-500 transition-transform duration-200 group-data-[state=open]/accordion:rotate-180" />
                </span>
              </AccordionTrigger>
              <AccordionContent className="pb-0">
                <div className="text-left text-[1rem] leading-relaxed">
                  {displayText ? (
                    <MarkdownContent content={displayText} />
                  ) : (
                    <span className="text-neutral-400">
                      Gathering thoughts...
                    </span>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </div>
    </div>
  );
}
