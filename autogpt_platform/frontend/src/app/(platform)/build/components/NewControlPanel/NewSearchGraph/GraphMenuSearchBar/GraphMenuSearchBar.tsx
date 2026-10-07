import { cn } from "@/lib/utils";
import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { useGraphMenuSearchBarComponent } from "./useGraphMenuSearchBarComponent";
import { Cancel01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      className={cn("flex min-h-14.25 items-center gap-2.5 px-4", className)}
    >
      <div className="flex h-6 w-6 items-center justify-center">
        <Icon
          icon={Search01Icon}
          className="h-6 w-6 text-zinc-700"
          strokeWidth={2}
        />
      </div>
      <input
        ref={inputRef}
        type="text"
        value={searchQuery}
        onChange={(e) => onSearchChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={"Search your graph for nodes, inputs, outputs..."}
        className="flex h-9 w-full bg-transparent p-0 font-sans text-base font-normal text-foreground outline-hidden placeholder:text-zinc-500"
        autoFocus
      />
      {searchQuery.length > 0 && (
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
