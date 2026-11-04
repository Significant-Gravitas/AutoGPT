"use client";

import type { Evidence } from "@/app/api/__generated__/models/evidence";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { useTaskEvidence } from "../../../../data/useTasksData";
import { attachmentHref, evidenceContent, evidenceIcon } from "../../helpers";
import { Fold } from "../Fold/Fold";
import { PanelMarkdown } from "../PanelMarkdown/PanelMarkdown";

interface Props {
  taskId: string;
  evidence: Evidence;
}

/** One piece of proof: what it shows, and the thing itself on demand. */
export function ProofLine({ taskId, evidence }: Props) {
  const [open, setOpen] = useState(false);
  const viewID = `proof-${evidence.id}`;

  return (
    <div className="flex flex-col">
      <div className="flex min-w-0 items-start gap-2">
        <Icon
          icon={evidenceIcon(evidence.kind)}
          tone="faint"
          size={13}
          className="mt-0.5 shrink-0"
          aria-hidden
        />
        <PanelMarkdown inline variant="small" tone="quiet">
          {evidence.summary || evidence.title}
        </PanelMarkdown>
        <Button
          variant="ghost-muted"
          size="inline-xs"
          aria-expanded={open}
          aria-controls={viewID}
          onClick={() => setOpen((v) => !v)}
          className="shrink-0"
        >
          {open ? "Hide" : "View"}
        </Button>
      </div>
      <Fold open={open} id={viewID}>
        <div className="pt-2 pl-5">
          <EvidenceView taskId={taskId} evidence={evidence} />
        </div>
      </Fold>
    </div>
  );
}

function EvidenceView({ taskId, evidence }: Props) {
  const { data, isLoading, isError } = useTaskEvidence(taskId, evidence.id);
  const content = evidenceContent(data);
  const href = attachmentHref(data?.attachment) ?? evidence.url ?? null;

  if (isLoading) return <Skeleton className="h-16 w-full" />;
  if (isError) {
    return (
      <Text variant="small" tone="error">
        This proof could not be loaded.
      </Text>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {content ? (
        <div className="max-h-64 overflow-auto rounded-control bg-surface-wash p-2.5">
          <PanelMarkdown variant="small" tone="body">
            {content}
          </PanelMarkdown>
        </div>
      ) : null}
      {href ? (
        <Link
          href={href}
          isExternal={!href.startsWith("/")}
          className="self-start"
        >
          {evidence.title || "Open"}
        </Link>
      ) : null}
      {!content && !href ? (
        <Text variant="small" tone="faint">
          {evidence.title}
        </Text>
      ) : null}
    </div>
  );
}
