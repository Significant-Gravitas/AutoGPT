"use client";

import { Button } from "@/components/ui/button";
import { CaretLeftIcon } from "@phosphor-icons/react";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";

interface Props {
  artifact: ArtifactRef;
  classification: ArtifactClassification;
  onExpand: () => void;
}

export function ArtifactRail({ artifact, classification, onExpand }: Props) {
  const Icon = classification.icon;
  return (
    <div className="flex h-full w-12 shrink-0 flex-col items-center gap-2 border-l border-l-[#80808017] bg-sidebar pt-2">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={onExpand}
        aria-label="Expand artifact panel"
        title={artifact.title}
      >
        <CaretLeftIcon className="!size-5" />
      </Button>
      <Icon aria-hidden className="size-5 text-zinc-400" />
    </div>
  );
}
