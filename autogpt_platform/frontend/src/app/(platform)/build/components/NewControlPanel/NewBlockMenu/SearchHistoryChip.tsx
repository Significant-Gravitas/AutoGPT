import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { ArrowUpRight } from "lucide-react";
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
    <Button
      variant="legacy-default"
      className={cn(
        "my-[1px] h-[2.25rem] space-x-1 rounded-[1.5rem] bg-surface-muted p-[0.375rem] pr-[0.625rem] shadow-none",
        // TODO(tokens): no fitting role for active:bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this is the pressed state (QUALITY 41); no fitting role for active:ring-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border colour) and foreground-disabled (a text colour), and this is a ring (QUALITY 41(b))
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300",
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
        // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41)
        "h-[2.25rem] w-32 rounded-[1.5rem] bg-zinc-100",
        className,
      )}
    />
  );
};

SearchHistoryChip.Skeleton = SearchHistoryChipSkeleton;
