"use client";
import React from "react";
import { BlockMenuSearchBar } from "../BlockMenuSearchBar/BlockMenuSearchBar";
import { Separator } from "@/components/atoms/Separator/Separator";
import { BlockMenuDefault } from "../BlockMenuDefault/BlockMenuDefault";
import { BlockMenuSearch } from "../BlockMenuSearch/BlockMenuSearch";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";

export const BlockMenuContent = () => {
  const searchQuery = useBlockMenuStore((state) => state.searchQuery);
  return (
    <div className="flex h-full w-full flex-col">
      <BlockMenuSearchBar />
      <Separator className="h-[1px] w-full text-foreground-disabled" />
      {searchQuery ? <BlockMenuSearch /> : <BlockMenuDefault />}
    </div>
  );
};
