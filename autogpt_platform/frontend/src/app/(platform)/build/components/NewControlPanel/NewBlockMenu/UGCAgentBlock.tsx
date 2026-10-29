import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { blockMenuButtonBase } from "./style";
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
    <button
      type="button"
      className={cn(
        blockMenuButtonBase,
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-xl bg-surface-muted p-2.5 pr-3.5 text-start",
        "hover:cursor-default hover:bg-surface-hover active:bg-surface-hover active:ring-1 active:ring-border-strong disabled:cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      {showImage && image_url && (
        <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-md bg-surface">
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
          <Text
            variant="body-medium"
            as="span"
            tone="emphasis"
            className={cn("line-clamp-1 group-disabled:text-foreground-subtle")}
          >
            {highlightText(title, highlightedText)}
          </Text>
        )}
        <div className="flex items-center space-x-1.5">
          {edited_time && (
            <Text
              variant="small-loose"
              as="span"
              tone="muted"
              className={cn(
                "line-clamp-1 group-disabled:text-foreground-subtle",
              )}
            >
              Edited {formatTimeAgo(edited_time.toISOString())}
            </Text>
          )}

          <Text variant="body-medium-tight" as="span" tone="subtle">
            •
          </Text>

          <Text
            variant="small-loose"
            as="span"
            tone="muted"
            className={cn("line-clamp-1 group-disabled:text-foreground-subtle")}
          >
            Version {version}
          </Text>

          <Text
            variant="small-medium-loose"
            as="span"
            tone="muted"
            className={cn("rounded-xl bg-surface-active px-2")}
          >
            Your Workflow
          </Text>
        </div>
      </div>
      <div
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded-lg bg-foreground-body group-disabled:bg-foreground-subtle",
        )}
      >
        {isLoading ? (
          <Icon
            icon={Loading03Icon}
            className="h-5 w-5 animate-spin text-action-foreground"
          />
        ) : (
          <Icon
            icon={PlusSignIcon}
            className="h-5 w-5 text-action-foreground"
            strokeWidth={2}
          />
        )}
      </div>
    </button>
  );
};

const UGCAgentBlockSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn(
        "flex h-[4.375rem] w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-xl bg-surface-hover p-2.5 pr-3.5",
        className,
      )}
    >
      <Skeleton className="h-[3.125rem] w-[5.625rem] rounded-md bg-surface-active" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-surface-active" />
        <div className="flex items-center gap-1">
          <Skeleton className="h-5 w-16 rounded bg-surface-active" />
          <Skeleton className="h-5 w-16 rounded bg-surface-active" />
        </div>
      </div>
      <Skeleton className="h-7 w-7 rounded-lg bg-surface-active" />
    </Skeleton>
  );
};

UGCAgentBlock.Skeleton = UGCAgentBlockSkeleton;
