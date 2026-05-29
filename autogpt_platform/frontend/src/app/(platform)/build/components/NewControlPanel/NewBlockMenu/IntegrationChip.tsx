import { Button } from "@/components/__legacy__/ui/button";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
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
      className={cn(
        "flex h-[3.25rem] w-full min-w-[7.5rem] justify-start gap-2 whitespace-normal rounded-[0.5rem] bg-zinc-50 p-2 pr-3 shadow-none",
        "hover:cursor-default hover:bg-zinc-100 focus:ring-0 active:bg-zinc-100 active:ring-1 active:ring-zinc-300",
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
        <span className="truncate font-sans text-sm font-normal leading-[1.375rem] text-zinc-800">
          {beautifyString(name)}
        </span>
      )}
    </Button>
  );
};

const IntegrationChipSkeleton: React.FC = () => {
  return (
    <Skeleton className="flex h-[3.25rem] w-full min-w-[7.5rem] gap-2 rounded-[0.5rem] bg-zinc-100 p-2 pr-3">
      <Skeleton className="h-9 w-12 rounded-[0.5rem] bg-zinc-200" />
      <Skeleton className="h-5 w-24 self-center rounded-sm bg-zinc-200" />
    </Skeleton>
  );
};

IntegrationChip.Skeleton = IntegrationChipSkeleton;
