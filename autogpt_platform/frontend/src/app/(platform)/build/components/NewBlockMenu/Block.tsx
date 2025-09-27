import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { beautifyString, cn } from "@/lib/utils";
import React, { ButtonHTMLAttributes } from "react";
import { highlightText } from "./helpers";
import { PlusIcon } from "@phosphor-icons/react";
interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  highlightedText?: string;
}

interface BlockComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const Block: BlockComponent = ({
  title,
  description,
  highlightedText,
  className,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem] text-start shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      <div className="flex flex-1 flex-col items-start gap-0.5">
        {title && (
          <span
            className={cn(
              "line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 group-disabled:text-zinc-400",
            )}
          >
            {highlightText(beautifyString(title), highlightedText)}
          </span>
        )}
        {description && (
          <span
            className={cn(
              "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
            )}
          >
            {highlightText(description, highlightedText)}
          </span>
        )}
      </div>
      <div
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        <PlusIcon className="h-5 w-5 text-zinc-50" />
      </div>
    </Button>
  );
};

const BlockSkeleton = () => {
  return (
    <Skeleton className="flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start space-x-3 rounded-[0.75rem] bg-zinc-100 px-[0.875rem] py-[0.625rem]">
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-zinc-200" />
        <Skeleton className="h-5 w-32 rounded bg-zinc-200" />
      </div>
      <Skeleton className="h-7 w-7 rounded-[0.5rem] bg-zinc-200" />
    </Skeleton>
  );
};

Block.Skeleton = BlockSkeleton;
