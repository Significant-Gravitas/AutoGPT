"use client";
import React from "react";
import BlockMenuSearchBar from "./BlockMenuSearchBar";
import BlockMenuSearch from "./search-and-filter//BlockMenuSearch";
import BlockMenuDefault from "./default/BlockMenuDefault";
import { Separator } from "@/components/ui/separator";
import { useBlockMenuContext } from "./block-menu-provider";

const BlockMenuContent: React.FC = () => {
  const { searchQuery } = useBlockMenuContext();
  return (
    <div className="flex h-full w-full flex-col">
      <BlockMenuSearchBar />
      <Separator className="h-[1px] w-full text-zinc-300" />
      {searchQuery ? <BlockMenuSearch /> : <BlockMenuDefault />}
    </div>
  );
};

export default BlockMenuContent;
