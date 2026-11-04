"use client";

import type { Contract } from "@/app/api/__generated__/models/contract";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { evidenceById } from "../../helpers";
import { ProofLine } from "../ProofLine/ProofLine";

interface Props {
  taskId: string;
  contract: Contract;
}

const LONG_RESULT = 160;

/** What came out of the task, with the main thing it made. */
export function ResultCard({ taskId, contract }: Props) {
  const [expanded, setExpanded] = useState(false);
  const result = contract.result;
  if (!result) return null;
  const primary = result.primary_evidence_id
    ? evidenceById(contract).get(result.primary_evidence_id)
    : undefined;
  const isLong = result.summary.length > LONG_RESULT;

  return (
    <section aria-label="Result" className="animate-inbox-enter px-1.5 py-4">
      <div className="flex flex-col gap-2 rounded-card bg-status-ok-surface p-4">
        <div className="flex items-center gap-1.5">
          <Icon icon={CheckmarkCircle02Icon} tone="ok" size={14} aria-hidden />
          <Text as="h3" variant="inbox-medium" tone="ok">
            Result
          </Text>
        </div>
        <Text
          variant="inbox"
          tone="ink"
          className={expanded ? "whitespace-pre-wrap" : "line-clamp-2"}
        >
          {result.summary}
        </Text>
        {isLong ? (
          <Button
            variant="ghost-muted"
            size="inline-xs"
            className="self-start"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Show less" : "Show all"}
          </Button>
        ) : null}
        {primary ? (
          <div className="rounded-control bg-surface p-2.5 shadow-card">
            <ProofLine taskId={taskId} evidence={primary} />
          </div>
        ) : null}
      </div>
    </section>
  );
}
