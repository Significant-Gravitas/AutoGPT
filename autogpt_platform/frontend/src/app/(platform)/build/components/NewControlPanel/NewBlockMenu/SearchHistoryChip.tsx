import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { blockMenuButtonBase } from "./style";
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
    <button
      type="button"
      className={cn(
        blockMenuButtonBase,
        "my-px h-[2.25rem] space-x-1 rounded-3xl bg-surface-muted p-1.5 pr-2.5",
        "hover:cursor-default hover:bg-surface-hover active:bg-surface-hover active:ring-1 active:ring-border-strong",
        className,
      )}
      {...rest}
    >
      <span className="flex text-foreground-tertiary">
        <Icon
          aria-hidden
          icon={ArrowUpRight01Icon}
          className="h-6 w-6"
          strokeWidth={1.25}
        />
      </span>
      <Text variant="body" as="span" tone="emphasis">
        {content}
      </Text>
    </button>
  );
};

const SearchHistoryChipSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn("h-[2.25rem] w-32 rounded-3xl bg-surface-hover", className)}
    />
  );
};

SearchHistoryChip.Skeleton = SearchHistoryChipSkeleton;
