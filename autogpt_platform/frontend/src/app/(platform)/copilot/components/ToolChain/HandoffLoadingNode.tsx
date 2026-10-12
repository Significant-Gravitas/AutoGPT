"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { WireNode } from "./WireNode";

interface Props {
  isLast: boolean;
  expertName: string;
}

/** The hand-off's card while the call is on its way: the approval card's
 *  outline in grey, so whichever card lands next takes its place. */
export function HandoffLoadingNode({ isLast, expertName }: Props) {
  return (
    <WireNode
      isLast={isLast}
      testId="handoff-loading-node"
      busy
      label={`Handing off to ${expertName}`}
    >
      <div className="flex items-center gap-3 border-b border-zinc-100 bg-zinc-50 px-4 py-3">
        <Skeleton className="size-7 shrink-0 rounded-full" />
        <Skeleton className="h-3 w-full max-w-[260px]" />
      </div>
      <div className="flex flex-col gap-2.5 px-4 py-3.5">
        <Skeleton className="h-3 w-12" />
        <Skeleton className="h-3 w-[92%]" />
        <Skeleton className="h-3 w-[70%]" />
        <div className="flex items-center gap-4">
          <Skeleton className="h-3 w-[140px]" />
          <Skeleton className="h-3 w-[110px]" />
          <Skeleton className="hidden h-3 w-40 sm:block" />
        </div>
      </div>
      <div className="flex gap-2 border-t border-zinc-100 px-4 py-3">
        <Skeleton className="h-9 w-[104px] rounded-full" />
        <Skeleton className="h-9 w-[88px] rounded-full" />
      </div>
    </WireNode>
  );
}
