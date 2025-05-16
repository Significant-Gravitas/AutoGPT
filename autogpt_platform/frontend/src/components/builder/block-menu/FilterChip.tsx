import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  number?: number;
  name?: string;
}

const FilterChip: React.FC<Props> = ({
  selected = false,
  number,
  name,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group w-fit space-x-1 rounded-[1.5rem] border border-zinc-300 bg-transparent px-[0.625rem] py-[0.375rem] shadow-none hover:bg-zinc-100 focus:ring-0 disabled:pointer-events-none",
      )}
      {...rest}
    >
      <span className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-600 group-disabled:text-zinc-400">
        {name}
      </span>
      {selected &&
        (number ? (
          <span className="flex h-[1.375rem] items-center rounded-[1.25rem] bg-violet-700 p-[0.375rem] text-zinc-50">
            {number}
          </span>
        ) : (
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-600">
            <X className="h-4 w-4 rounded-full text-white" strokeWidth={1.25} />
          </span>
        ))}
    </Button>
  );
};

export default FilterChip;
