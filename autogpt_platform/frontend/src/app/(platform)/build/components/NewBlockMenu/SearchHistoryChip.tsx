import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { ArrowUpRight } from "lucide-react";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  content?: string;
}

interface SearchHistoryChipComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const SearchHistoryChip: SearchHistoryChipComponent = ({
  content,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "my-[1px] h-[2.25rem] space-x-1 rounded-[1.5rem] bg-zinc-50 p-[0.375rem] pr-[0.625rem] shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300",
        className,
      )}
      {...rest}
    >
      <ArrowUpRight className="h-6 w-6 text-zinc-500" strokeWidth={1.25} />
      <span className="font-sans text-sm font-normal leading-[1.375rem] text-zinc-800">
        {content}
      </span>
    </Button>
  );
};

const SearchHistoryChipSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn("h-[2.25rem] w-32 rounded-[1.5rem] bg-zinc-100", className)}
    />
  );
};

SearchHistoryChip.Skeleton = SearchHistoryChipSkeleton;