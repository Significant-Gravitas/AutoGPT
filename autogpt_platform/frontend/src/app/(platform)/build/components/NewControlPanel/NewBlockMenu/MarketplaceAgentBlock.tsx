import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { cn } from "@/lib/utils";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";
import Link from "next/link";
import { highlightText } from "./helpers";
import {
  ArrowSquareOutIcon,
  CircleNotchIcon,
  PlusIcon,
} from "@phosphor-icons/react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  creator_name?: string;
  number_of_runs?: number;
  image_url?: string;
  highlightedText?: string;
  slug: string;
  loading: boolean;
}

interface MarketplaceAgentBlockComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const MarketplaceAgentBlock: MarketplaceAgentBlockComponent = ({
  title,
  image_url,
  creator_name,
  number_of_runs,
  className,
  loading,
  highlightedText,
  slug,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-zinc-50 p-[0.625rem] pr-[0.875rem] text-start shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-[0.375rem] bg-white">
        {image_url && (
          <Image
            src={image_url}
            alt="integration-icon"
            fill
            sizes="5.625rem"
            className="w-full object-contain group-disabled:opacity-50"
          />
        )}
      </div>
      <div className="flex flex-1 flex-col items-start gap-0.5">
        {title && (
          <span
            className={cn(
              "line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800 group-disabled:text-zinc-400",
            )}
          >
            {highlightText(title, highlightedText)}
          </span>
        )}
        <div className="flex items-center space-x-2.5">
          <span
            className={cn(
              "truncate font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
            )}
          >
            By {creator_name}
          </span>

          <span className="font-sans text-zinc-400">•</span>

          <span
            className={cn(
              "truncate font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
            )}
          >
            {number_of_runs} runs
          </span>
          <span className="font-sans text-zinc-400">•</span>
          <Link
            href={`/marketplace/agent/${creator_name}/${slug}`}
            className="flex gap-0.5 truncate"
            onClick={(e) => e.stopPropagation()}
          >
            <span className="font-sans text-xs leading-5 text-blue-700 underline">
              Agent page
            </span>
            <ArrowSquareOutIcon
              className="h-4 w-4 text-blue-700"
              strokeWidth={1}
            />
          </Link>
        </div>
      </div>
      <div
        className={cn(
          "flex h-7 min-w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        {!loading ? (
          <PlusIcon className="h-5 w-5 text-zinc-50" strokeWidth={2} />
        ) : (
          <CircleNotchIcon className="h-5 w-5 animate-spin" />
        )}
      </div>
    </Button>
  );
};

const MarketplaceAgentBlockSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn(
        "flex h-[4.375rem] w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-[0.75rem] bg-zinc-100 p-[0.625rem] pr-[0.875rem]",
        className,
      )}
    >
      <Skeleton className="h-[3.125rem] w-[5.625rem] rounded-[0.375rem] bg-zinc-200" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-zinc-200" />
        <div className="flex items-center gap-1">
          <Skeleton className="h-5 w-16 rounded bg-zinc-200" />

          <Skeleton className="h-5 w-16 rounded bg-zinc-200" />
        </div>
      </div>
      <Skeleton className="h-7 w-7 rounded-[0.5rem] bg-zinc-200" />
    </Skeleton>
  );
};

MarketplaceAgentBlock.Skeleton = MarketplaceAgentBlockSkeleton;
