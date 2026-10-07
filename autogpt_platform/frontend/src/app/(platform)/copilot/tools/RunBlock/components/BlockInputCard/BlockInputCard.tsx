"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";

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
      <Button
        variant="ghost"
        size="xs"
        unmask={false}
        className="h-auto px-0 font-normal text-muted-foreground hover:border-transparent hover:bg-transparent hover:text-foreground"
        onClick={() => setExpanded((prev) => !prev)}
      >
        {expanded ? "Hide inputs" : `Show inputs (${entries.length})`}
      </Button>
      {expanded && (
        <ContentGrid className="mt-2 mb-2">
          {entries.map(([key, value]) => (
            <ContentCard key={key}>
              <ContentCardTitle className="text-xs">{key}</ContentCardTitle>
              <pre className="mt-1 max-h-48 overflow-auto text-xs wrap-break-word whitespace-pre-wrap text-muted-foreground">
                {renderValue(value)}
              </pre>
            </ContentCard>
          ))}
        </ContentGrid>
      )}
    </div>
  );
}
