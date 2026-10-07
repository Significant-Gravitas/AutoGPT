import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
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
      variant="ghost"
      size="md"
      unmask={false}
      className={cn(
        "my-px h-9 min-w-0 gap-1 rounded-3xl bg-zinc-50 p-1.5 pr-2.5 shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:opacity-50",
        className,
      )}
      {...rest}
    >
      <Icon
        icon={ArrowUpRight01Icon}
        className="h-6 w-6 text-zinc-500"
        strokeWidth={1.25}
      />
      <span className="font-sans text-sm leading-5.5 font-normal text-zinc-800">
        {content}
      </span>
    </Button>
  );
};

const SearchHistoryChipSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton className={cn("h-9 w-32 rounded-3xl bg-zinc-100", className)} />
  );
};

SearchHistoryChip.Skeleton = SearchHistoryChipSkeleton;
