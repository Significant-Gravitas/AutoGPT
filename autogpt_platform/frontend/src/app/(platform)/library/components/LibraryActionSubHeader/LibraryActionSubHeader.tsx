"use client";

import LibrarySortMenu from "../LibrarySortMenu/LibrarySortMenu";

interface LibraryActionSubHeaderProps {
  agentCount: number;
}

export default function LibraryActionSubHeader({
  agentCount,
}: LibraryActionSubHeaderProps) {
  return (
    <div className="flex items-center justify-between pb-[10px]">
      <div className="flex items-center gap-[10px] p-2">
        <span className="font-poppin w-[96px] text-[18px] font-semibold leading-[28px] text-neutral-800">
          My agents
        </span>
        <span
          className="w-[70px] font-sans text-[14px] font-normal leading-6"
          data-testid="agents-count"
        >
          {agentCount} agents
        </span>
      </div>
      <LibrarySortMenu />
    </div>
  );
}
