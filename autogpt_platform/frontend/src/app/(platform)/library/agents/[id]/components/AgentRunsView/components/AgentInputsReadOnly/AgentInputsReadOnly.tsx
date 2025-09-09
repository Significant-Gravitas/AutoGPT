"use client";

import React from "react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

type Props = {
  agent: LibraryAgent;
  inputs?: Record<string, any> | null;
};

function getAgentInputFields(agent: LibraryAgent): Record<string, any> {
  const schema = agent.input_schema as unknown as {
    properties?: Record<string, any>;
  } | null;
  if (!schema || !schema.properties) return {};
  const properties = schema.properties as Record<string, any>;
  const visibleEntries = Object.entries(properties).filter(
    ([, sub]) => !sub?.hidden,
  );
  return Object.fromEntries(visibleEntries);
}

function renderValue(value: any): string {
  if (value === undefined || value === null) return "";
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  )
    return String(value);
  try {
    return JSON.stringify(value, undefined, 2);
  } catch {
    return String(value);
  }
}

export function AgentInputsReadOnly({ agent, inputs }: Props) {
  const fields = getAgentInputFields(agent);
  const entries = Object.entries(fields);

  if (!inputs || entries.length === 0) {
    return <div className="text-neutral-600">No input for this run.</div>;
  }

  return (
    <div className="flex flex-col gap-4">
      {entries.map(([key, sub]) => (
        <div key={key} className="flex flex-col gap-1.5">
          <label className="text-sm font-medium">{sub?.title || key}</label>
          <p className="whitespace-pre-wrap break-words text-sm text-neutral-700">
            {renderValue((inputs as Record<string, any>)[key])}
          </p>
        </div>
      ))}
    </div>
  );
}
