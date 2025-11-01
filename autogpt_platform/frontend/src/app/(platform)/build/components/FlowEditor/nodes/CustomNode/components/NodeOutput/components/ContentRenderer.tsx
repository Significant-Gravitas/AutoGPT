"use client";

import { globalRegistry } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import type { OutputMetadata } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";

export const ContentRenderer: React.FC<{
  value: any;
}> = ({ value }) => {
  const metadata: OutputMetadata = {};

  if (typeof value === "object" && value !== null) {
    if (value.type) metadata.type = value.type;
    if (value.mimeType) metadata.mimeType = value.mimeType;
    if (value.filename) metadata.filename = value.filename;
  }

  const renderer = globalRegistry.getRenderer(value, metadata);

  console.log(renderer);

  return (
    <div className="[&>*]:rounded-xlarge [&>*]:!text-xs">
      {renderer?.render(value, metadata)}
    </div>
  );
};
