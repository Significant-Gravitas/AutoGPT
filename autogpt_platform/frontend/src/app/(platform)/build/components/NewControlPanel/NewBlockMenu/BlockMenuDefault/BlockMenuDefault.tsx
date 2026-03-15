import React from "react";

import { Separator } from "@/components/__legacy__/ui/separator";
import { BlockMenuDefaultContent } from "../BlockMenuDefaultContent/BlockMenuDefaultContent";
import { BlockMenuSidebar } from "../BlockMenuSidebar/BlockMenuSidebar";

export const BlockMenuDefault = () => {
  return (
    <div className="flex flex-1 overflow-y-auto">
      <BlockMenuSidebar />
      <Separator className="h-full w-[1px] text-zinc-300" />
      <BlockMenuDefaultContent />
    </div>
  );
};
