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
        "group flex h-16 w-full min-w-[7.5rem] items-center justify-start space-x-3 rounded-xl bg-surface-muted px-3.5 py-2.5 text-start whitespace-normal",
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
              variant="body"
              weight="medium"
              tone={rest.disabled ? "subtle" : "body"}
              className="line-clamp-1 flex-1"
            >
              {beautifyString(title)}
            </Text>
          )}
          <span className="flex h-[1.375rem] w-[1.6875rem] items-center justify-center rounded-pill bg-surface-hover p-1.5">
            <Text
              variant="body"
              weight="medium"
              as="span"
              tone={rest.disabled ? "subtle" : "muted"}
              className="block"
            >
              {number_of_blocks}
            </Text>
          </span>
        </div>
        <Text
          variant="small"
          leading="5"
          as="span"
          tone={rest.disabled ? "subtle" : "muted"}
          className="line-clamp-1"
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
        "flex h-16 w-full min-w-[7.5rem] items-center justify-start rounded-xl",
        className,
      )}
    >
      <div className="flex w-full items-center justify-start space-x-3 px-3.5 py-2.5">
        <Skeleton
          variant="nested"
          className="h-[2.625rem] w-[2.625rem] rounded-lg"
        />
        <div className="flex flex-1 flex-col items-start gap-0.5">
          <div className="flex w-full items-center justify-between">
            <Skeleton variant="nested" className="h-[1.375rem] w-24 rounded" />
            <Skeleton
              variant="nested"
              className="h-[1.375rem] w-[1.6875rem] rounded-pill"
            />
          </div>
          <Skeleton variant="nested" className="h-5 w-[80%] rounded" />
        </div>
      </div>
    </Skeleton>
  );
};

Integration.Skeleton = IntegrationSkeleton;
