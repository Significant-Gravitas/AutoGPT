"use client";

import { Button } from "@/components/ui/button";
import { CaretLeftIcon } from "@phosphor-icons/react";
import type { ArtifactRef } from "../../../store";

interface Props {
  artifact: ArtifactRef;
  onExpand: () => void;
}

export function ArtifactRail({ artifact, onExpand }: Props) {
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
    </div>
  );
}
