import { cn } from "@/lib/utils";
import React from "react";
import { Input } from "@/components/__legacy__/ui/input";
import { useBlockMenuSearchBar } from "./useBlockMenuSearchBar";
import { Button } from "@/components/atoms/Button/Button";
import { Cancel01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      data-id="blocks-control-search-bar"
      className={cn("flex min-h-14.25 items-center gap-2.5 px-4", className)}
    >
      <div className="flex h-6 w-6 items-center justify-center">
        <Icon
          icon={Search01Icon}
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
          "m-0 border-none p-0 font-sans text-base font-normal text-zinc-800 shadow-none outline-hidden",
          "placeholder:text-zinc-400 focus:shadow-none focus:ring-0 focus:outline-hidden",
        )}
      />
      {localQuery.length > 0 && (
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Clear search"
          withTooltip={false}
          onClick={handleClear}
          className="w-auto hover:border-transparent hover:bg-transparent"
        >
          <Icon
            icon={Cancel01Icon}
            className="h-6 w-6 text-zinc-700"
            strokeWidth={2}
          />
        </Button>
      )}
    </div>
  );
};
