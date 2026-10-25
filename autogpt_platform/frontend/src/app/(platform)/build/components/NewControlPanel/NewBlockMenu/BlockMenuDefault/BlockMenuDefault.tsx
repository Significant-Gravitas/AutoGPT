import React from "react";

import { Separator } from "@/components/atoms/Separator/Separator";
import { BlockMenuDefaultContent } from "../BlockMenuDefaultContent/BlockMenuDefaultContent";
import { BlockMenuSidebar } from "../BlockMenuSidebar/BlockMenuSidebar";

export const BlockMenuDefault = () => {
  return (
    <div className="flex flex-1 overflow-y-auto">
      <BlockMenuSidebar />
      <Separator
        // TODO(tokens): no fitting role for text-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (disabled text) and border-strong (a border colour), and this is a static divider (QUALITY 41(c))
        className="h-full w-[1px] text-zinc-300"
      />
      <BlockMenuDefaultContent />
    </div>
  );
};
