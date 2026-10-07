import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import React, { ButtonHTMLAttributes } from "react";
import { highlightText } from "./helpers";
import { formatTimeAgo } from "@/lib/utils/time";
import { Loading03Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useImageFallback } from "@/hooks/useImageFallback";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading?: boolean;
  title?: string;
  edited_time?: Date;
  version?: number;
  image_url: string | null;
  highlightedText?: string;
}

interface UGCAgentBlockComponent extends React.FC<Props> {
  Skeleton: React.FC<{ className?: string }>;
}

export const UGCAgentBlock: UGCAgentBlockComponent = ({
  isLoading,
  title,
  image_url,
  edited_time = new Date(),
  version,
  className,
  highlightedText,
  ...rest
}) => {
  const { showImage, handleImageError } = useImageFallback(image_url);

  return (
    <Button
      variant="ghost"
      unmask={false}
      className={cn(
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-xl bg-zinc-50 p-2.5 pr-3.5 text-start shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...rest}
    >
      {showImage && image_url && (
        <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-md bg-white">
          <Image
            src={image_url}
            unoptimized={isLocalStoreMediaUrl(image_url)}
            alt="integration-icon"
            fill
            sizes="5.625rem"
            className="w-full object-contain group-disabled:opacity-50"
            onError={handleImageError}
          />
        </div>
      )}
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
        <div className="flex items-center space-x-1.5">
          {edited_time && (
            <span
              className={cn(
                "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
              )}
            >
              Edited {formatTimeAgo(edited_time.toISOString())}
            </span>
          )}

          <span className="font-sans text-zinc-400">•</span>

          <span
            className={cn(
              "line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500 group-disabled:text-zinc-400",
            )}
          >
            Version {version}
          </span>

          <span
            className={cn(
              "rounded-xl bg-zinc-200 px-2 font-sans text-xs leading-5 text-zinc-500",
            )}
          >
            Your Agent
          </span>
        </div>
      </div>
      <div
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        {isLoading ? (
          <Icon
            icon={Loading03Icon}
            className="h-5 w-5 animate-spin text-zinc-50"
          />
        ) : (
          <Icon
            icon={PlusSignIcon}
            className="h-5 w-5 text-zinc-50"
            strokeWidth={2}
          />
        )}
      </div>
    </Button>
  );
};

const UGCAgentBlockSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn(
        "flex h-[4.375rem] w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-xl bg-zinc-100 p-2.5 pr-3.5",
        className,
      )}
    >
      <Skeleton className="h-[3.125rem] w-[5.625rem] rounded-md bg-zinc-200" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-zinc-200" />
        <div className="flex items-center gap-1">
          <Skeleton className="h-5 w-16 rounded bg-zinc-200" />
          <Skeleton className="h-5 w-16 rounded bg-zinc-200" />
        </div>
      </div>
      <Skeleton className="h-7 w-7 rounded-lg bg-zinc-200" />
    </Skeleton>
  );
};

UGCAgentBlock.Skeleton = UGCAgentBlockSkeleton;
