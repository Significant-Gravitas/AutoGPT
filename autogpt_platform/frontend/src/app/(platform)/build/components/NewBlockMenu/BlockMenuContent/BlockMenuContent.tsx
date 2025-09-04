"use client";
import React from "react";
import { useBlockMenuContext } from "../block-menu-provider";
import { BlockMenuSearchBar } from "../BlockMenuSearchBar/BlockMenuSearchBar";
import { Separator } from "@/components/ui/separator";
import { BlockMenuDefault } from "../BlockMenuDefault/BlockMenuDefault";
import { BlockMenuSearch } from "../BlockMenuSearch/BlockMenuSearch";

export const BlockMenuContent = () => {
  const { searchQuery } = useBlockMenuContext();
  return (
    <div className="flex h-full w-full flex-col">
      <BlockMenuSearchBar />
      <Separator className="h-[1px] w-full text-zinc-300" />
      {searchQuery ? <BlockMenuSearch /> : <BlockMenuDefault />}
    </div>
  );
};
