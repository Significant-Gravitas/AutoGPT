"use client";

import { ChatCircleDotsIcon } from "@phosphor-icons/react";

interface Props {
  children: React.ReactNode;
}

export function ReasoningCollapse({ children }: Props) {
  return (
    <div className="my-1 flex items-start gap-1.5 text-xs text-zinc-500">
      <ChatCircleDotsIcon size={14} weight="bold" className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1">
        <span className="font-medium">Reasoning:</span>{" "}
        <span className="[&_pre]:m-0 [&_pre]:inline [&_pre]:whitespace-pre-wrap [&_pre]:bg-transparent [&_pre]:p-0 [&_pre]:text-xs [&_pre]:text-zinc-500">
          {children}
        </span>
      </div>
    </div>
  );
}
