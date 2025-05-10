import React from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export function Chip({
  children,
  className,
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <Badge
      className={cn(
        "rounded-[30px] border border-zinc-400 bg-white px-3.5 py-2 text-center font-sans text-sm font-normal text-zinc-600 shadow-none hover:border-zinc-500 hover:bg-zinc-100 md:text-base",
        className,
      )}
    >
      {children}
    </Badge>
  );
}
