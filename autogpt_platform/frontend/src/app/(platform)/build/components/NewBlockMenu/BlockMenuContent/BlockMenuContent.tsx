"use client";
import React from "react";
import { BlockMenuSearchBar } from "../BlockMenuSearchBar/BlockMenuSearchBar";
import { Separator } from "@/components/__legacy__/ui/separator";
import { BlockMenuDefault } from "../BlockMenuDefault/BlockMenuDefault";
import { BlockMenuSearch } from "../BlockMenuSearch/BlockMenuSearch";
import { useBlockMenuStore } from "../../../stores/blockMenuStore";

export const BlockMenuContent = () => {
  const { searchQuery } = useBlockMenuStore();
  return (
    <div className="flex h-full w-full flex-col">
      <BlockMenuSearchBar />
      <Separator className="h-[1px] w-full text-zinc-300" />
      {searchQuery ? <BlockMenuSearch /> : <BlockMenuDefault />}
    </div>
  );
};
