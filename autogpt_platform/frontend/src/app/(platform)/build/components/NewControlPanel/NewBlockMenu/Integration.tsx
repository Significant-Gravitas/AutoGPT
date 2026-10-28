import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
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
      variant="legacy-default"
      className={cn(
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-[0.75rem] bg-surface-muted px-[0.875rem] py-[0.625rem] text-start shadow-none",
        // TODO(tokens): no fitting role for active:ring-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border colour) and foreground-disabled (a text colour), and this is a ring (QUALITY 41(b))
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-surface-muted active:ring-1 active:ring-zinc-300 disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="relative h-[2.625rem] w-[2.625rem] overflow-hidden rounded-[0.5rem] bg-surface">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            sizes="2.25rem"
            className="w-full rounded-[0.5rem] object-contain group-disabled:opacity-50"
          />
        )}
      </div>

      <div className="w-full">
        <div className="flex items-center justify-between gap-2">
          {title && (
            <Text
              variant="body-medium"
              tone="body"
              className="line-clamp-1 flex-1 group-disabled:text-foreground-subtle"
            >
              {beautifyString(title)}
            </Text>
          )}
          <Text
            variant="body-medium"
            as="span"
            tone="muted"
            // TODO(tokens): no exact role: bg-[#f0f0f0] #F0F0F0
            className="flex h-[1.375rem] w-[1.6875rem] items-center justify-center rounded-[1.25rem] bg-[#f0f0f0] p-1.5 group-disabled:text-foreground-subtle"
          >
            {number_of_blocks}
          </Text>
        </div>
        <Text
          variant="small-loose"
          as="span"
          tone="muted"
          className="line-clamp-1 group-disabled:text-foreground-subtle"
        >
          {description}
        </Text>
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
        // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41)
        "flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start space-x-3 rounded-[0.75rem] bg-zinc-100 px-[0.875rem] py-[0.625rem]",
        className,
      )}
    >
      <Skeleton
        // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
        className="h-[2.625rem] w-[2.625rem] rounded-[0.5rem] bg-zinc-200"
      />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <div className="flex w-full items-center justify-between">
          <Skeleton
            // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
            className="h-[1.375rem] w-24 rounded bg-zinc-200"
          />
          <Skeleton
            // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
            className="h-[1.375rem] w-[1.6875rem] rounded-[1.25rem] bg-zinc-200"
          />
        </div>
        <Skeleton
          // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
          className="h-5 w-[80%] rounded bg-zinc-200"
        />
      </div>
    </Skeleton>
  );
};

Integration.Skeleton = IntegrationSkeleton;
