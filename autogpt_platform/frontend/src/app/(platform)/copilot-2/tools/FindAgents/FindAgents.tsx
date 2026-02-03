"use client";

import { ToolUIPart } from "ai";
import { CaretDownIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useState } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  getAgentHref,
  getAnimationText,
  getFindAgentsOutput,
  getSourceLabelFromToolType,
  StateIcon,
} from "./helpers";
import { cn } from "@/lib/utils";

export interface FindAgentsToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: FindAgentsToolPart;
}

export function FindAgentsTool({ part }: Props) {
  const text = getAnimationText(part);
  const output = getFindAgentsOutput(part);
  const shouldReduceMotion = useReducedMotion();
  const [isExpanded, setIsExpanded] = useState(false);

  const query =
    typeof part.input === "object" && part.input !== null
      ? String((part.input as { query?: unknown }).query ?? "").trim()
      : "";

  const isAgentsFound =
    part.state === "output-available" && output?.type === "agents_found";
  const hasAgents =
    isAgentsFound &&
    output.agents.length > 0 &&
    (typeof output.count !== "number" || output.count > 0);
  const totalCount = isAgentsFound ? output.count : 0;
  const { label: sourceLabel, source } = getSourceLabelFromToolType(part.type);
  const scopeText =
    source === "library"
      ? "in your library"
      : source === "marketplace"
        ? "in marketplace"
        : "";

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasAgents && (
        <div className="mt-2 rounded-2xl border bg-background px-3 py-2">
          <button
            type="button"
            aria-expanded={isExpanded}
            onClick={() => setIsExpanded((v) => !v)}
            className="flex w-full items-center justify-between gap-3 py-1 text-left"
          >
            <div className="flex min-w-0 items-center gap-2">
              <span className="rounded-full border bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                {sourceLabel}
              </span>
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-foreground">
                  Agent results
                </p>
                <p className="truncate text-xs text-muted-foreground">
                  Found {totalCount} {scopeText}
                  {query ? ` for "${query}"` : ""}
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
                <div className="grid gap-2 pb-2 pt-3 sm:grid-cols-2">
                  {output.agents.map((agent) => {
                    const href = getAgentHref(agent);
                    const agentSource =
                      agent.source === "library"
                        ? "Library"
                        : agent.source === "marketplace"
                          ? "Marketplace"
                          : null;
                    return (
                      <div
                        key={agent.id}
                        className="rounded-2xl border bg-background p-3"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <p className="truncate text-sm font-medium text-foreground">
                                {agent.name}
                              </p>
                              {agentSource && (
                                <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                                  {agentSource}
                                </span>
                              )}
                            </div>
                            <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                              {agent.description}
                            </p>
                          </div>
                          {href && (
                            <Link
                              href={href}
                              className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                            >
                              Open
                            </Link>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
