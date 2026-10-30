import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { blockMenuButtonBase } from "./style";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import React, { ButtonHTMLAttributes } from "react";
import Link from "next/link";
import { highlightText } from "./helpers";
import {
  LinkSquare01Icon,
  Loading03Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useImageFallback } from "@/hooks/useImageFallback";

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
  const { showImage, handleImageError } = useImageFallback(image_url);

  return (
    <button
      type="button"
      className={cn(
        blockMenuButtonBase,
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 rounded-xl bg-surface-muted p-2.5 pr-3.5 text-start whitespace-normal",
        "hover:cursor-default hover:bg-surface-hover active:bg-surface-hover active:ring-1 active:ring-border-strong disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-md bg-surface">
        {showImage && image_url && (
          <Image
            src={image_url}
            unoptimized={isLocalStoreMediaUrl(image_url)}
            alt="integration-icon"
            fill
            sizes="5.625rem"
            className="w-full object-contain group-disabled:opacity-50"
            onError={handleImageError}
          />
        )}
      </div>
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
        <div className="flex items-center space-x-2.5">
          <Text
            variant="small-loose"
            as="span"
            tone="muted"
            className={cn("truncate group-disabled:text-foreground-subtle")}
          >
            By {creator_name}
          </Text>

          <Text variant="body-medium-tight" as="span" tone="subtle">
            •
          </Text>

          <Text
            variant="small-loose"
            as="span"
            tone="muted"
            className={cn("truncate group-disabled:text-foreground-subtle")}
          >
            {number_of_runs} runs
          </Text>
          <Text variant="body-medium-tight" as="span" tone="subtle">
            •
          </Text>
          <Link
            href={`/marketplace/agent/${creator_name}/${slug}`}
            className="flex gap-0.5 truncate"
            onClick={(e) => e.stopPropagation()}
          >
            <Text
              variant="small-medium-loose-underline"
              as="span"
              className="text-info-foreground"
            >
              Workflow page
            </Text>
            <Icon
              icon={LinkSquare01Icon}
              className="h-4 w-4 text-info-foreground"
              strokeWidth={1}
            />
          </Link>
        </div>
      </div>
      <div
        className={cn(
          "flex h-7 min-w-7 items-center justify-center rounded-lg bg-foreground-body group-disabled:bg-foreground-subtle",
        )}
      >
        {!loading ? (
          <Icon
            icon={PlusSignIcon}
            className="h-5 w-5 text-action-foreground"
            strokeWidth={2}
          />
        ) : (
          <Icon icon={Loading03Icon} className="h-5 w-5 animate-spin" />
        )}
      </div>
    </button>
  );
};

const MarketplaceAgentBlockSkeleton: React.FC<{ className?: string }> = ({
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

MarketplaceAgentBlock.Skeleton = MarketplaceAgentBlockSkeleton;
