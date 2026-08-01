"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export interface RecipientOption {
  id: string | null;
  name: string;
  avatarUrl: string | null;
}

interface Props {
  recipient: RecipientOption;
  options: RecipientOption[];
  onSelect: (id: string | null) => void;
  /** True while the expert list is still loading behind a `?expertId=` deep
   * link — showing the Autopilot fallback there would name the wrong
   * recipient. */
  isLoading?: boolean;
}

const CHIP_CLASSNAME =
  "ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium text-zinc-700 shadow-sm";

export function RecipientChip({
  recipient,
  options,
  onSelect,
  isLoading,
}: Props) {
  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Loading recipient"
        className={CHIP_CLASSNAME}
      >
        <Skeleton className="h-5 w-5 rounded-full" />
        <Skeleton className="h-3 w-16" />
      </div>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`Sending to ${recipient.name} — change recipient`}
          className={cn(
            CHIP_CLASSNAME,
            "transition-colors hover:bg-neutral-50",
          )}
        >
          <RecipientAvatar option={recipient} />
          {recipient.name}
          <Icon icon={ArrowDown01Icon} className="size-3 text-zinc-400" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {options.map((option) => (
          <DropdownMenuItem
            key={option.id ?? "autopilot"}
            onClick={() => onSelect(option.id)}
            className={cn("gap-2", option.id === recipient.id && "bg-zinc-100")}
          >
            <RecipientAvatar option={option} />
            {option.name}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function RecipientAvatar({ option }: { option: RecipientOption }) {
  return (
    <Avatar className="h-5 w-5">
      {option.avatarUrl ? (
        <AvatarImage src={option.avatarUrl} alt={option.name} />
      ) : null}
      <AvatarFallback className="text-[9px]">
        {option.name.slice(0, 2)}
      </AvatarFallback>
    </Avatar>
  );
}
