import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import React, { ButtonHTMLAttributes, useState } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  number?: number;
  name?: string;
}

export const FilterChip: React.FC<Props> = ({
  selected = false,
  number,
  name,
  className,
  ...rest
}) => {
  const [isHovering, setIsHovering] = useState(false);
  return (
    <Button
      className={cn(
        "group w-fit space-x-1 rounded-[1.5rem] border border-zinc-300 bg-transparent px-[0.625rem] py-[0.375rem] shadow-none transition-transform duration-300 ease-in-out",
        "hover:border-violet-500 hover:bg-transparent focus:ring-0 disabled:pointer-events-none",
        selected && "border-0 bg-violet-700 hover:border",
      )}
      {...rest}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <span
        className={cn(
          "font-sans text-sm font-medium leading-[1.375rem] text-zinc-600 group-hover:text-zinc-600 group-disabled:text-zinc-400",
          selected && "text-zinc-50",
        )}
      >
        {name}
      </span>
      {selected &&
        (isHovering && number !== undefined ? (
          <span className="flex h-[1.375rem] items-center rounded-[1.25rem] bg-violet-700 p-[0.375rem] text-zinc-50 transition-all duration-300 ease-in-out animate-in fade-in zoom-in">
            {number > 100 ? "100+" : number}
          </span>
        ) : (
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-zinc-50 transition-all duration-300 ease-in-out">
            <X
              className="h-3 w-3 rounded-full text-violet-700"
              strokeWidth={2}
            />
          </span>
        ))}
    </Button>
  );
};


