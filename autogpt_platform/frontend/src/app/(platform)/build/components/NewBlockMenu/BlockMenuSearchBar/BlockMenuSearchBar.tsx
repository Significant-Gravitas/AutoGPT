import { cn } from "@/lib/utils";
import React from "react";
import { Input } from "@/components/ui/input";
import { useBlockMenuSearchBar } from "./useBlockMenuSearchBar";
import { Button } from "@/components/ui/button";
import { MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react";

interface BlockMenuSearchBarProps {
  className?: string;
}

export const BlockMenuSearchBar: React.FC<BlockMenuSearchBarProps> = ({
  className = "",
}) => {
  const {
    handleClear,
    inputRef,
    localQuery,
    setLocalQuery,
    debouncedSetSearchQuery,
  } = useBlockMenuSearchBar();

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
        value={localQuery}
        onChange={(e) => {
          setLocalQuery(e.target.value);
          debouncedSetSearchQuery(e.target.value);
        }}
        placeholder={"Blocks, Agents, Integrations or Keywords..."}
        className={cn(
          "m-0 border-none p-0 font-sans text-base font-normal text-zinc-800 shadow-none outline-none",
          "placeholder:text-zinc-400 focus:shadow-none focus:outline-none focus:ring-0",
        )}
      />
      {localQuery.length > 0 && (
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
