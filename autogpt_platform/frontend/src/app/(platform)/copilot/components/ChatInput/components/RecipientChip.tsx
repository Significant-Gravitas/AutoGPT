"use client";

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
import { ExpertAvatar } from "../../ChatMessagesContainer/components/ExpertAvatar/ExpertAvatar";

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
  "ml-2 inline-flex h-9 items-center gap-1.5 rounded-2xl border border-neutral-200 bg-white pl-1.5 pr-2 text-sm font-medium text-zinc-700 shadow-sm";

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
        <Skeleton className="h-6 w-6 rounded-full" />
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
            "group transition-colors hover:bg-neutral-50",
          )}
        >
          <RecipientAvatar option={recipient} />
          {recipient.name}
          <Icon
            icon={ArrowDown01Icon}
            className="size-3.5 text-zinc-600 transition-transform duration-150 group-data-[state=open]:rotate-180"
          />
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

// The same face the expert wears in the thread: Autopilot's own avatar for
// the null recipient, the generated one for experts without an upload.
function RecipientAvatar({ option }: { option: RecipientOption }) {
  return (
    <ExpertAvatar
      name={option.name}
      avatarUrl={option.avatarUrl}
      isAutopilot={option.id === null}
      size="sm"
    />
  );
}
