import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { cn } from "@/lib/utils";
import { ArrowUpRight } from "lucide-react";
import React, { ButtonHTMLAttributes } from "react";
import { Text } from "@/components/atoms/Text/Text";

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
      className={cn(
        "my-[1px] h-[2.25rem] space-x-1 rounded-[1.5rem] bg-surface-muted p-[0.375rem] pr-[0.625rem] shadow-none",
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-surface-hover active:ring-1 active:ring-border-strong",
        className,
      )}
      {...rest}
    >
      <ArrowUpRight
        className="h-6 w-6 text-foreground-tertiary"
        strokeWidth={1.25}
      />
      <Text variant="body" as="span" tone="emphasis">
        {content}
      </Text>
    </Button>
  );
};

const SearchHistoryChipSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn(
        "h-[2.25rem] w-32 rounded-[1.5rem] bg-surface-hover",
        className,
      )}
    />
  );
};

SearchHistoryChip.Skeleton = SearchHistoryChipSkeleton;
