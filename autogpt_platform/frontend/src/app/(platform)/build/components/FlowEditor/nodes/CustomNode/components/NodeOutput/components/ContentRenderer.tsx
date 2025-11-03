"use client";

import { globalRegistry } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import type { OutputMetadata } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";

export const TextRenderer: React.FC<{
  value: any;
  truncateLengthLimit?: number;
}> = ({ value, truncateLengthLimit = 100 }) => {
  const text =
    typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  const truncated =
    truncateLengthLimit && text.length > truncateLengthLimit
      ? text.slice(0, truncateLengthLimit) + "..."
      : text;

  return <div className="break-words bg-zinc-50 p-3 text-xs">{truncated}</div>;
};

export const ContentRenderer: React.FC<{
  value: any;
  shortContent?: boolean;
}> = ({ value, shortContent = false }) => {
  const metadata: OutputMetadata = {};

  if (typeof value === "object" && value !== null) {
    if (value.type) metadata.type = value.type;
    if (value.mimeType) metadata.mimeType = value.mimeType;
    if (value.filename) metadata.filename = value.filename;
  }

  const renderer = globalRegistry.getRenderer(value, metadata);

  if (
    renderer?.name === "ImageRenderer" ||
    renderer?.name === "VideoRenderer" ||
    !shortContent
  ) {
    return (
      <div className="[&>*]:rounded-xlarge [&>*]:!text-xs">
        {renderer?.render(value, metadata)}
      </div>
    );
  }

  return (
    <div className="[&>*]:rounded-xlarge [&>*]:!text-xs">
      <TextRenderer value={value} truncateLengthLimit={100} />
    </div>
  );
};
