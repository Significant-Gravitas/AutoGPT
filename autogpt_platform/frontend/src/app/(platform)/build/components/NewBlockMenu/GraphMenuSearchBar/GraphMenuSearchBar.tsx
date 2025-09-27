import { cn } from "@/lib/utils";
import React from "react";
import { Input } from "@/components/__legacy__/ui/input";
import { Button } from "@/components/__legacy__/ui/button";
import { MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react";
import { useGraphMenuSearchBarComponent } from "./useGraphMenuSearchBarComponent";

interface GraphMenuSearchBarProps {
  className?: string;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
}

export const GraphMenuSearchBar: React.FC<GraphMenuSearchBarProps> = ({
  className = "",
  searchQuery,
  onSearchChange,
  onKeyDown,
}) => {
  const { inputRef, handleClear } = useGraphMenuSearchBarComponent({
    onSearchChange,
  });

  return (
    <div
      className={cn(
        "flex min-h-[3.5625rem] items-center gap-2.5 px-4",
        className,
      )}
    >
      <div className="flex h-6 w-6 items-center justify-center">
        <MagnifyingGlassIcon
          className="h-6 w-6 text-zinc-700"
          strokeWidth={2}
        />
      </div>
      <Input
        ref={inputRef}
        type="text"
        value={searchQuery}
        onChange={(e) => onSearchChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={"Search your graph for nodes, inputs, outputs..."}
        className={cn(
          "m-0 border-none p-0 font-sans text-base font-normal text-zinc-800 shadow-none outline-none",
          "placeholder:text-zinc-400 focus:shadow-none focus:outline-none focus:ring-0",
        )}
        autoFocus
      />
      {searchQuery.length > 0 && (
        <Button
          variant="ghost"
          size={"sm"}
          onClick={handleClear}
          className="p-0 hover:bg-transparent"
        >
          <XIcon className="h-6 w-6 text-zinc-700" strokeWidth={2} />
        </Button>
      )}
    </div>
  );
};
