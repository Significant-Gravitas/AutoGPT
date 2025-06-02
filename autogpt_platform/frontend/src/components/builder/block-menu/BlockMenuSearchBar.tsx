import { cn } from "@/lib/utils";
import { Search, X } from "lucide-react";
import React, { useRef, useState, useEffect, useMemo } from "react";
import { useBlockMenuContext } from "./block-menu-provider";
import { Button } from "@/components/ui/button";
import debounce from "lodash/debounce";
import { Input } from "@/components/ui/input";

interface BlockMenuSearchBarProps {
  className?: string;
}

const BlockMenuSearchBar: React.FC<BlockMenuSearchBarProps> = ({
  className = "",
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [localQuery, setLocalQuery] = useState("");
  const { searchQuery, setSearchQuery, searchId, setSearchId, setFilters } =
    useBlockMenuContext();

  const debouncedSetSearchQuery = useMemo(
    () =>
      debounce((value: string) => {
        setSearchQuery(value);
        if (value.length === 0) {
          setSearchId(undefined);
        } else if (!searchId) {
          setSearchId(crypto.randomUUID());
        }
      }, 500),
    [setSearchQuery, setSearchId, searchId],
  );

  useEffect(() => {
    return () => {
      debouncedSetSearchQuery.cancel();
    };
  }, [debouncedSetSearchQuery]);

  const handleClear = () => {
    setLocalQuery("");
    setSearchQuery("");
    setSearchId(undefined);
    setFilters({
      categories: {
        blocks: false,
        integrations: false,
        marketplace_agents: false,
        my_agents: false,
        providers: false,
      },
      createdBy: [],
    });
    debouncedSetSearchQuery.cancel();
  };

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
          size="sm"
          onClick={handleClear}
          className="p-0 hover:bg-transparent"
        >
          <X className="h-6 w-6 text-zinc-700" strokeWidth={2} />
        </Button>
      )}
    </div>
  );
};

export default BlockMenuSearchBar;
