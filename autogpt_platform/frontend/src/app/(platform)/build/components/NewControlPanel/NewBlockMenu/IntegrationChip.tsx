import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
import { blockMenuButtonBase } from "./style";
import Image from "next/image";
import React, { ButtonHTMLAttributes } from "react";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  name?: string;
  icon_url?: string;
}

interface IntegrationChipComponent extends React.FC<Props> {
  Skeleton: React.FC;
}

export const IntegrationChip: IntegrationChipComponent = ({
  icon_url,
  name,
  className,
  ...rest
}) => {
  return (
    <button
      type="button"
      className={cn(
        blockMenuButtonBase,
        "flex h-[3.25rem] w-full min-w-[7.5rem] justify-start gap-2 whitespace-normal rounded-lg bg-surface-muted p-2 pr-3",
        "hover:cursor-default hover:bg-surface-hover active:bg-surface-hover active:ring-1 active:ring-border-strong",
        className,
      )}
      {...rest}
    >
      <div className="relative h-9 w-9 rounded-lg bg-transparent">
        {icon_url && (
          <Image
            src={icon_url}
            alt="integration-icon"
            fill
            sizes="2.25rem"
            className="w-full object-contain"
          />
        )}
      </div>
      {name && (
        <Text variant="body" as="span" tone="emphasis" className="truncate">
          {beautifyString(name)}
        </Text>
      )}
    </button>
  );
};

const IntegrationChipSkeleton: React.FC = () => {
  return (
    <Skeleton className="flex h-[3.25rem] w-full min-w-[7.5rem] gap-2 rounded-lg bg-surface-hover p-2 pr-3">
      <Skeleton className="h-9 w-12 rounded-lg bg-surface-active" />
      <Skeleton className="h-5 w-24 self-center rounded-sm bg-surface-active" />
    </Skeleton>
  );
};

IntegrationChip.Skeleton = IntegrationChipSkeleton;
