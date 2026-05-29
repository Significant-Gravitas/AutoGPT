"use client";

import { ArrowsOutSimple } from "@phosphor-icons/react";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";

interface Props {
  artifact: ArtifactRef;
  classification: ArtifactClassification;
  onExpand: () => void;
}

export function ArtifactMinimizedStrip({
  artifact,
  classification,
  onExpand,
}: Props) {
  const Icon = classification.icon;

  return (
    <div className="flex h-full w-10 flex-col items-center border-l border-zinc-200 bg-white pt-3">
      <button
        type="button"
        onClick={onExpand}
        className="rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        title="Expand panel"
      >
        <ArrowsOutSimple size={16} />
      </button>
      <div className="mt-3 text-zinc-400">
        <Icon size={16} />
      </div>
      <span
        className="mt-2 text-xs text-zinc-400"
        style={{
          writingMode: "vertical-rl",
          textOrientation: "mixed",
          maxHeight: "120px",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {artifact.title}
      </span>
    </div>
  );
}
