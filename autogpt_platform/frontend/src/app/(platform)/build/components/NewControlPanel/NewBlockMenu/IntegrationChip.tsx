import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { beautifyString, cn } from "@/lib/utils";
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
    <Button
      variant="legacy-default"
      className={cn(
        "flex h-[3.25rem] w-full min-w-[7.5rem] justify-start gap-2 whitespace-normal rounded-[0.5rem] bg-surface-muted p-2 pr-3 shadow-none",
        // TODO(tokens): no fitting role for active:bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this is the pressed state (QUALITY 41); no fitting role for active:ring-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border colour) and foreground-disabled (a text colour), and this is a ring (QUALITY 41(b))
        "hover:cursor-default hover:bg-surface-hover focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300",
        className,
      )}
      {...rest}
    >
      <div className="relative h-9 w-9 rounded-[0.5rem] bg-transparent">
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
    </Button>
  );
};

const IntegrationChipSkeleton: React.FC = () => {
  return (
    <Skeleton
      // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41)
      className="flex h-[3.25rem] w-full min-w-[7.5rem] gap-2 rounded-[0.5rem] bg-zinc-100 p-2 pr-3"
    >
      <Skeleton
        // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
        className="h-9 w-12 rounded-[0.5rem] bg-zinc-200"
      />
      <Skeleton
        // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
        className="h-5 w-24 self-center rounded-sm bg-zinc-200"
      />
    </Skeleton>
  );
};

IntegrationChip.Skeleton = IntegrationChipSkeleton;
