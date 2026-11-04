"use client";

import type { Evidence } from "@/app/api/__generated__/models/evidence";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { PanelMarkdown } from "../PanelMarkdown/PanelMarkdown";
import { ProofLine } from "../ProofLine/ProofLine";

interface Props {
  taskId: string;
  summary: string | null;
  evidence: Evidence[];
}

const LONG_RESULT = 280;

/** The outcome, shown once: what came out of the task and its proof. */
export function ResultCard({ taskId, summary, evidence }: Props) {
  const [expanded, setExpanded] = useState(false);
  if (!summary && !evidence.length) return null;
  const isLong = (summary?.length ?? 0) > LONG_RESULT;

  return (
    <section aria-label="Outcome" className="animate-inbox-enter px-1.5 py-4">
      <div className="flex flex-col gap-2 rounded-card bg-status-ok-surface p-4">
        <div className="flex items-center gap-1.5">
          <Icon icon={CheckmarkCircle02Icon} tone="ok" size={14} aria-hidden />
          <Text as="h3" variant="inbox-medium" tone="ok">
            Outcome
          </Text>
        </div>
        {summary ? (
          <div
            className={cn(
              "relative",
              isLong && !expanded && "max-h-32 overflow-hidden",
            )}
          >
            <PanelMarkdown>{summary}</PanelMarkdown>
          </div>
        ) : null}
        {isLong ? (
          <Button
            variant="ghost-muted"
            size="inline-xs"
            className="self-start"
            aria-expanded={expanded}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Show less" : "Show all"}
          </Button>
        ) : null}
        {evidence.length ? (
          <div className="flex flex-col gap-1.5 rounded-control bg-surface p-2.5 shadow-card">
            {evidence.map((item) => (
              <ProofLine key={item.id} taskId={taskId} evidence={item} />
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}
