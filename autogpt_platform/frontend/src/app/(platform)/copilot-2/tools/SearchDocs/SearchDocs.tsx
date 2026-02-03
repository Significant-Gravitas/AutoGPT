"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { useMemo, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { CaretDownIcon } from "@phosphor-icons/react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { cn } from "@/lib/utils";
import {
  getDocsToolOutput,
  getDocsToolTitle,
  getToolLabel,
  getAnimationText,
  StateIcon,
  toDocsUrl,
  type DocsToolType,
} from "./helpers";

export interface DocsToolPart {
  type: DocsToolType;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: DocsToolPart;
}

function truncate(text: string, maxChars: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, maxChars).trimEnd()}…`;
}

export function SearchDocsTool({ part }: Props) {
  const shouldReduceMotion = useReducedMotion();
  const [isExpanded, setIsExpanded] = useState(false);

  const output = getDocsToolOutput(part);
  const text = getAnimationText(part);

  const normalized = useMemo(() => {
    if (!output) return null;
    const title = getDocsToolTitle(part.type, output);
    const label = getToolLabel(part.type);
    return { title, label };
  }, [output, part.type]);

  const isOutputAvailable = part.state === "output-available" && !!output;

  const hasExpandableContent =
    isOutputAvailable &&
    ((output.type === "doc_search_results" && output.count > 0) ||
      output.type === "doc_page" ||
      output.type === "no_results" ||
      output.type === "error");

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasExpandableContent && normalized && (
        <div className="mt-2 rounded-2xl border bg-background px-3 py-2">
          <button
            type="button"
            aria-expanded={isExpanded}
            onClick={() => setIsExpanded((v) => !v)}
            className="flex w-full items-center justify-between gap-3 py-1 text-left"
          >
            <div className="flex min-w-0 items-center gap-2">
              <span className="rounded-full border bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                {normalized.label}
              </span>
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-foreground">
                  {normalized.title}
                </p>
                <p className="truncate text-xs text-muted-foreground">
                  {output.type === "doc_search_results"
                    ? `Found ${output.count} result${output.count === 1 ? "" : "s"} for "${output.query}"`
                    : output.type === "doc_page"
                      ? output.path
                      : output.message}
                </p>
              </div>
            </div>
            <CaretDownIcon
              className={cn(
                "h-4 w-4 shrink-0 text-muted-foreground transition-transform",
                isExpanded && "rotate-180",
              )}
              weight="bold"
            />
          </button>

          <AnimatePresence initial={false}>
            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0, filter: "blur(10px)" }}
                animate={{ height: "auto", opacity: 1, filter: "blur(0px)" }}
                exit={{ height: 0, opacity: 0, filter: "blur(10px)" }}
                transition={
                  shouldReduceMotion
                    ? { duration: 0 }
                    : { type: "spring", bounce: 0.35, duration: 0.55 }
                }
                className="overflow-hidden"
                style={{ willChange: "height, opacity, filter" }}
              >
                {output.type === "doc_search_results" && (
                  <div className="grid gap-2 pb-2 pt-3">
                    {output.results.map((r) => {
                      const href = r.doc_url ?? toDocsUrl(r.path);
                      return (
                        <div
                          key={r.path}
                          className="rounded-2xl border bg-background p-3"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium text-foreground">
                                {r.title}
                              </p>
                              <p className="mt-0.5 truncate text-xs text-muted-foreground">
                                {r.path}
                                {r.section ? ` • ${r.section}` : ""}
                              </p>
                              <p className="mt-2 text-xs text-muted-foreground">
                                {truncate(r.snippet, 240)}
                              </p>
                            </div>
                            <Link
                              href={href}
                              target="_blank"
                              rel="noreferrer"
                              className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                            >
                              Open
                            </Link>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {output.type === "doc_page" && (
                  <div className="pb-2 pt-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-foreground">
                          {output.title}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-muted-foreground">
                          {output.path}
                        </p>
                      </div>
                      <Link
                        href={output.doc_url ?? toDocsUrl(output.path)}
                        target="_blank"
                        rel="noreferrer"
                        className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                      >
                        Open
                      </Link>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                      {truncate(output.content, 800)}
                    </p>
                  </div>
                )}

                {output.type === "no_results" && (
                  <div className="pb-2 pt-3">
                    <p className="text-sm text-foreground">{output.message}</p>
                    {output.suggestions && output.suggestions.length > 0 && (
                      <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted-foreground">
                        {output.suggestions.slice(0, 5).map((s) => (
                          <li key={s}>{s}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}

                {output.type === "error" && (
                  <div className="pb-2 pt-3">
                    <p className="text-sm text-foreground">{output.message}</p>
                    {output.error && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        {output.error}
                      </p>
                    )}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
