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
      <Separator
        // TODO(tokens): no fitting role for text-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (disabled text) and border-strong (a border colour), and this is a static divider (QUALITY 41(c))
        className="h-[1px] w-full text-zinc-300"
      />
      {searchQuery ? <BlockMenuSearch /> : <BlockMenuDefault />}
    </div>
  );
};
