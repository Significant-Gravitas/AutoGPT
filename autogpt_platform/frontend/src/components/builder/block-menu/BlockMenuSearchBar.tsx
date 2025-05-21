import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Search } from "lucide-react";
import React, { useRef } from "react";
import { useBlockMenuContext } from "./block-menu-provider";

interface BlockMenuSearchBarProps {
  className?: string;
}

const BlockMenuSearchBar: React.FC<BlockMenuSearchBarProps> = ({
  className = "",
}) => {
  const inputRef = useRef(null);
  const { searchQuery, setSearchQuery } = useBlockMenuContext();
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
