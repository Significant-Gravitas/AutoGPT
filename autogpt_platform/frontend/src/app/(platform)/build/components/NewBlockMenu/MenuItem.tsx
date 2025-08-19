// BLOCK MENU TODO: We need to add a better hover state to it; currently it's not in the design either.

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  number?: number;
  name?: string;
}

export const MenuItem: React.FC<Props> = ({
  selected = false,
  number,
  name,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "flex h-[2.375rem] w-[12.875rem] justify-between whitespace-normal rounded-[0.5rem] bg-transparent p-2 pl-3 shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0",
        selected && "bg-zinc-100",
        className,
      )}
      {...rest}
    >
      <span className="truncate font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
        {name}
      </span>
      {number && (
        <span className="font-sans text-sm font-normal leading-[1.375rem] text-zinc-600">
          {number > 100 ? "100+" : number}
        </span>
      )}
    </Button>
  );
};