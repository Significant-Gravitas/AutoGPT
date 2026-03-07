"use client";

import { LightbulbIcon } from "@phosphor-icons/react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  children: React.ReactNode;
}

export function ReasoningCollapse({ children }: Props) {
  return (
    <Dialog title="Reasoning">
      <Dialog.Trigger>
        <button
          type="button"
          className="flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-700"
        >
          <LightbulbIcon size={12} weight="bold" />
          <span>Show reasoning</span>
        </button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="space-y-1">{children}</div>
      </Dialog.Content>
    </Dialog>
  );
}
