"use client";
import React, { useState } from "react";
import BlockMenuSearchBar from "./BlockMenuSearchBar";
import BlockMenuSearch from "./search-and-filter//BlockMenuSearch";
import BlockMenuDefault from "./default/BlockMenuDefault";
import { Separator } from "@/components/ui/separator";

const BlockMenuContent: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState("");
  return (
    <div className="flex h-full w-full flex-col">
      {/* Search Bar */}
      <BlockMenuSearchBar
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
      />

      <Separator className="h-[1px] w-full text-zinc-300" />

      {/* Content */}
      {/* BLOCK MENU TODO : search after 3 characters */}
      {searchQuery ? <BlockMenuSearch /> : <BlockMenuDefault />}
    </div>
  );
};

export default BlockMenuContent;
