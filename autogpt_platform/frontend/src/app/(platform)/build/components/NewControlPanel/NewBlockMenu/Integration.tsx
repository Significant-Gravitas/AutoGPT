import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { beautifyString, cn } from "@/lib/utils";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  title?: string;
  description?: string;
  icon_url?: string;
  number_of_blocks?: number;
}

interface IntegrationComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const Integration: IntegrationComponent = ({
  title,
  icon_url,
  description,
  className,
  number_of_blocks,
  ...rest
}) => {
  return (
    <Button
      className={cn(
        "group flex h-16 w-full min-w-30 items-center justify-start space-x-3 rounded-[0.75rem] bg-zinc-50 px-3.5 py-2.5 text-start whitespace-normal shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-50 active:ring-1 active:ring-zinc-300 disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="relative h-10.5 w-10.5 overflow-hidden rounded-small bg-white">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            sizes="2.25rem"
            className="w-full rounded-small object-contain group-disabled:opacity-50"
          />
        )}
      </div>

      <div className="w-full">
        <div className="flex items-center justify-between gap-2">
          {title && (
            <p className="line-clamp-1 flex-1 font-sans text-sm leading-5.5 font-medium text-zinc-700 group-disabled:text-zinc-400">
              {beautifyString(title)}
            </p>
          )}
          <span className="flex h-5.5 w-6.75 items-center justify-center rounded-xlarge bg-[#f0f0f0] p-1.5 font-sans text-sm leading-5.5 text-zinc-500 group-disabled:text-zinc-400">
            {number_of_blocks}
          </span>
        </div>
        <span className="line-clamp-1 font-sans text-xs leading-5 font-normal text-zinc-500 group-disabled:text-zinc-400">
          {description}
        </span>
      </div>
    </Button>
  );
};

const IntegrationSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn(
        "flex h-16 w-full min-w-30 animate-pulse items-center justify-start space-x-3 rounded-[0.75rem] bg-zinc-100 px-3.5 py-2.5",
        className,
      )}
    >
      <Skeleton className="h-10.5 w-10.5 rounded-small bg-zinc-200" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <div className="flex w-full items-center justify-between">
          <Skeleton className="h-5.5 w-24 rounded bg-zinc-200" />
          <Skeleton className="h-5.5 w-6.75 rounded-xlarge bg-zinc-200" />
        </div>
        <Skeleton className="h-5 w-[80%] rounded bg-zinc-200" />
      </div>
    </Skeleton>
  );
};

Integration.Skeleton = IntegrationSkeleton;
