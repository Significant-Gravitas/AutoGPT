"use client";

import type {
  OutputMetadata,
  OutputRenderer,
} from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import {
  globalRegistry,
  OutputActions,
  OutputItem,
} from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import React, { useMemo } from "react";

type OutputsRecord = Record<string, Array<unknown>>;

interface RunOutputsProps {
  outputs: OutputsRecord;
}

export function RunOutputs({ outputs }: RunOutputsProps) {
  const items = useMemo(() => {
    const list: Array<{
      key: string;
      label: string;
      value: unknown;
      metadata?: OutputMetadata;
      renderer: OutputRenderer;
    }> = [];

    Object.entries(outputs || {}).forEach(([key, values]) => {
      (values || []).forEach((value, index) => {
        const metadata: OutputMetadata = {};
        if (
          typeof value === "object" &&
          value !== null &&
          !React.isValidElement(value)
        ) {
          const obj = value as Record<string, unknown>;
          if (typeof obj["type"] === "string")
            metadata.type = obj["type"] as string;
          if (typeof obj["mimeType"] === "string")
            metadata.mimeType = obj["mimeType"] as string;
          if (typeof obj["filename"] === "string")
            metadata.filename = obj["filename"] as string;
          if (typeof obj["language"] === "string")
            metadata.language = obj["language"] as string;
        }

        const renderer = globalRegistry.getRenderer(value, metadata);
        if (renderer) {
          list.push({
            key: `${key}-${index}`,
            label: index === 0 ? key : "",
            value,
            metadata,
            renderer,
          });
        } else {
          const textRenderer = globalRegistry
            .getAllRenderers()
            .find((r) => r.name === "TextRenderer");
          if (textRenderer) {
            list.push({
              key: `${key}-${index}`,
              label: index === 0 ? key : "",
              value:
                typeof value === "string"
                  ? value
                  : JSON.stringify(value, null, 2),
              metadata,
              renderer: textRenderer,
            });
          }
        }
      });
    });

    return list;
  }, [outputs]);

  if (!items.length) {
    return <div className="text-neutral-600">No output from this run.</div>;
  }

  return (
    <div className="relative flex flex-col gap-4">
      <div className="absolute -top-3 right-0 z-10">
        <OutputActions
          items={items.map((item) => ({
            value: item.value,
            metadata: item.metadata,
            renderer: item.renderer,
          }))}
        />
      </div>
      {items.map((item) => (
        <OutputItem
          key={item.key}
          value={item.value}
          metadata={item.metadata}
          renderer={item.renderer}
          label={item.label}
        />
      ))}
    </div>
  );
}
