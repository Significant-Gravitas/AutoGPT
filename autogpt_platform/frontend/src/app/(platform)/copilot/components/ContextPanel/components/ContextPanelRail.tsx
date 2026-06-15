"use client";

import { Button } from "@/components/ui/button";
import { CaretRightIcon, FolderIcon } from "@phosphor-icons/react";

interface Props {
  onExpand: () => void;
}

export function ContextPanelRail({ onExpand }: Props) {
  return (
    <div className="flex h-full w-12 shrink-0 flex-col items-center gap-2 border-l border-l-[#80808017] bg-sidebar pt-2">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={onExpand}
        aria-label="Expand workspace panel"
      >
        <CaretRightIcon className="!size-5" />
      </Button>
      <FolderIcon aria-hidden className="size-5 text-zinc-400" />
    </div>
  );
}
