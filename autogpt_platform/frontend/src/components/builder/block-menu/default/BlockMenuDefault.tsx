import React from "react";
import { BlockMenuSidebar } from "./BlockMenuSidebar";
import { Separator } from "@/components/ui/separator";
import { BlockMenuDefaultContent } from "./BlockMenuDefaultContent";

export const BlockMenuDefault = () => {
  return (
    <div className="flex flex-1 overflow-y-auto">
      <BlockMenuSidebar />
      <Separator className="h-full w-[1px] text-zinc-300" />
      <BlockMenuDefaultContent />
    </div>
  );
};
