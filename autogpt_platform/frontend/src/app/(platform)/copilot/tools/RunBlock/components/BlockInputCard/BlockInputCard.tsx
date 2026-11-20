"use client";

import { useState } from "react";

import {
  ContentCard,
  ContentCardTitle,
  ContentGrid,
} from "../../../../components/ToolAccordion/AccordionContent";

interface Props {
  inputData: Record<string, unknown>;
}

function renderValue(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

export function BlockInputCard({ inputData }: Props) {
  const [expanded, setExpanded] = useState(false);
  const entries = Object.entries(inputData);

  if (entries.length === 0) return null;

  return (
    <div>
      <button
        className="text-caption text-foreground-tertiary transition-colors hover:text-foreground-base"
        onClick={() => setExpanded((prev) => !prev)}
      >
        {expanded ? "Hide inputs" : `Show inputs (${entries.length})`}
      </button>
      {expanded && (
        <ContentGrid className="mt-2 mb-2">
          {entries.map(([key, value]) => (
            <ContentCard key={key}>
              <ContentCardTitle size="small">{key}</ContentCardTitle>
              <pre className="mt-1 max-h-48 overflow-auto text-caption break-words whitespace-pre-wrap text-foreground-tertiary">
                {renderValue(value)}
              </pre>
            </ContentCard>
          ))}
        </ContentGrid>
      )}
    </div>
  );
}
