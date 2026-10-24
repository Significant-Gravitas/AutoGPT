import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
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
import { Text } from "@/components/atoms/Text/Text";

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
    <Button
      className={cn(
        "group flex h-[4.375rem] w-full min-w-[7.5rem] items-center justify-start gap-3 whitespace-normal rounded-[0.75rem] bg-surface-muted p-[0.625rem] pr-[0.875rem] text-start shadow-none",
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-surface-hover active:ring-1 active:ring-border-strong disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="relative h-[3.125rem] w-[5.625rem] overflow-hidden rounded-[0.375rem] bg-surface">
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
            variant="small"
            as="span"
            tone="muted"
            className={cn(
              "truncate leading-5 group-disabled:text-foreground-subtle",
            )}
          >
            By {creator_name}
          </Text>

          <span className="font-sans text-foreground-subtle">•</span>

          <Text
            variant="small"
            as="span"
            tone="muted"
            className={cn(
              "truncate leading-5 group-disabled:text-foreground-subtle",
            )}
          >
            {number_of_runs} runs
          </Text>
          <span className="font-sans text-foreground-subtle">•</span>
          <Link
            href={`/marketplace/agent/${creator_name}/${slug}`}
            className="flex gap-0.5 truncate"
            onClick={(e) => e.stopPropagation()}
          >
            <span
              // TODO(tokens): no exact role: text-blue-700 #1D4ED8
              className="font-sans text-xs leading-5 text-blue-700 underline"
            >
              Workflow page
            </span>
            <Icon
              icon={LinkSquare01Icon}
              // TODO(tokens): no exact role: text-blue-700 #1D4ED8
              className="h-4 w-4 text-blue-700"
              strokeWidth={1}
            />
          </Link>
        </div>
      </div>
      <div
        className={cn(
          // TODO(tokens): no fitting role for bg-zinc-700 #505057; #505057 is foreground-body (a text colour); no fitting role for group-disabled:bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
          "flex h-7 min-w-7 items-center justify-center rounded-[0.5rem] bg-zinc-700 group-disabled:bg-zinc-400",
        )}
      >
        {!loading ? (
          <Icon
            icon={PlusSignIcon}
            // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
            className="h-5 w-5 text-zinc-50"
            strokeWidth={2}
          />
        ) : (
          <Icon icon={Loading03Icon} className="h-5 w-5 animate-spin" />
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

MarketplaceAgentBlock.Skeleton = MarketplaceAgentBlockSkeleton;
