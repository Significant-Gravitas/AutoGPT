import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Search, X } from "lucide-react";
import React, { useRef, useState } from "react";
import { useBlockMenuContext } from "./block-menu-provider";
import { Button } from "@/components/ui/button";

interface BlockMenuSearchBarProps {
  className?: string;
}

const BlockMenuSearchBar: React.FC<BlockMenuSearchBarProps> = ({
  className = "",
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const { searchQuery, setSearchQuery, searchId, setSearchId } =
    useBlockMenuContext();

  return (
    <div
      className={cn(
        "flex min-h-[3.5625rem] items-center gap-2.5 px-4",
        className,
      )}
    >
      <Search className="h-6 w-6 text-zinc-700" strokeWidth={2} />
      <Input
        ref={inputRef}
        type="text"
        value={searchQuery}
        onChange={(e) => {
          setSearchQuery(e.target.value);
          if (e.target.value.length === 0) {
            setSearchId(undefined);
          } else if (!searchId) {
            setSearchId(crypto.randomUUID());
          }
        }}
        placeholder={"Blocks, Agents, Integrations or Keywords..."}
        className={cn(
          "m-0 border-none p-0 font-sans text-base font-normal text-zinc-800 shadow-none outline-none",
          "placeholder:text-zinc-400 focus:shadow-none focus:outline-none focus:ring-0",
        )}
      />
    </div>
  );
};

export default BlockMenuSearchBar;
