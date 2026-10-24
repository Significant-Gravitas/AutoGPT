import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import React, { ButtonHTMLAttributes } from "react";
import { highlightText } from "./helpers";
import { formatTimeAgo } from "@/lib/utils/time";
import { Loading03Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useImageFallback } from "@/hooks/useImageFallback";
import { Text } from "@/components/atoms/Text/Text";

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
      className={cn(
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-surface-muted p-[0.625rem] pr-[0.875rem] text-start shadow-none",
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-surface-hover active:ring-1 active:ring-border-strong disabled:cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      {showImage && image_url && (
        <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-[0.375rem] bg-surface">
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
            className={cn(
              "line-clamp-1 text-foreground-emphasis group-disabled:text-foreground-subtle",
            )}
          >
            {highlightText(title, highlightedText)}
          </Text>
        )}
        <div className="flex items-center space-x-1.5">
          {edited_time && (
            <Text
              variant="small"
              as="span"
              className={cn(
                "line-clamp-1 leading-5 text-foreground-tertiary group-disabled:text-foreground-subtle",
              )}
            >
              Edited {formatTimeAgo(edited_time.toISOString())}
            </Text>
          )}

          <span className="font-sans text-foreground-subtle">•</span>

          <Text
            variant="small"
            as="span"
            className={cn(
              "line-clamp-1 leading-5 text-foreground-tertiary group-disabled:text-foreground-subtle",
            )}
          >
            Version {version}
          </Text>

          <span
            className={cn(
              "rounded-[0.75rem] bg-surface-active px-[0.5rem] font-sans text-xs leading-5 text-foreground-tertiary",
            )}
          >
            Your Workflow
          </span>
        </div>
      </div>
      <div
        className={cn(
          // TODO(tokens): no fitting role for bg-zinc-700 #505057; #505057 is foreground-body (a text colour); no fitting role for group-disabled:bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
          "flex h-7 w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        {isLoading ? (
          <Icon
            icon={Loading03Icon}
            // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
            className="h-5 w-5 animate-spin text-zinc-50"
          />
        ) : (
          <Icon
            icon={PlusSignIcon}
            // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
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
        "flex h-[4.375rem] w-full min-w-[7.5rem] animate-pulse items-center justify-start gap-3 rounded-[0.75rem] bg-surface-hover p-[0.625rem] pr-[0.875rem]",
        className,
      )}
    >
      <Skeleton className="h-[3.125rem] w-[5.625rem] rounded-[0.375rem] bg-surface-active" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <Skeleton className="h-[1.375rem] w-24 rounded bg-surface-active" />
        <div className="flex items-center gap-1">
          <Skeleton className="h-5 w-16 rounded bg-surface-active" />
          <Skeleton className="h-5 w-16 rounded bg-surface-active" />
        </div>
      </div>
      <Skeleton className="h-7 w-7 rounded-[0.5rem] bg-surface-active" />
    </Skeleton>
  );
};

UGCAgentBlock.Skeleton = UGCAgentBlockSkeleton;
