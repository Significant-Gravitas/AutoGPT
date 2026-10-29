import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import { blockMenuButtonBase } from "./style";
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
    <button
      type="button"
      className={cn(
        blockMenuButtonBase,
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 whitespace-normal rounded-xl bg-surface-muted px-3.5 py-2.5 text-start",
        "hover:cursor-default hover:bg-surface-hover active:bg-surface-muted active:ring-1 active:ring-border-strong disabled:pointer-events-none",
        className,
      )}
      {...rest}
    >
      <div className="relative h-[2.625rem] w-[2.625rem] overflow-hidden rounded-lg bg-surface">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            sizes="2.25rem"
            className="w-full rounded-lg object-contain group-disabled:opacity-50"
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
            className="flex h-[1.375rem] w-[1.6875rem] items-center justify-center rounded-full bg-surface-hover p-1.5 group-disabled:text-foreground-subtle"
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
    </button>
  );
};

const IntegrationSkeleton: React.FC<{ className?: string }> = ({
  className,
}) => {
  return (
    <Skeleton
      className={cn(
        "flex h-16 w-full min-w-[7.5rem] animate-pulse items-center justify-start space-x-3 rounded-xl bg-surface-hover px-3.5 py-2.5",
        className,
      )}
    >
      <Skeleton className="h-[2.625rem] w-[2.625rem] rounded-lg bg-surface-active" />
      <div className="flex flex-1 flex-col items-start gap-0.5">
        <div className="flex w-full items-center justify-between">
          <Skeleton className="h-[1.375rem] w-24 rounded bg-surface-active" />
          <Skeleton className="h-[1.375rem] w-[1.6875rem] rounded-full bg-surface-active" />
        </div>
        <Skeleton className="h-5 w-[80%] rounded bg-surface-active" />
      </div>
    </Skeleton>
  );
};

Integration.Skeleton = IntegrationSkeleton;
