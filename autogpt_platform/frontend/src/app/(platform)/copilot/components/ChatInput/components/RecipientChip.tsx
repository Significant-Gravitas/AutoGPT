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
import { cn } from "@/lib/utils";
import { CaretDownIcon } from "@phosphor-icons/react";

export interface RecipientOption {
  id: string | null;
  name: string;
  avatarUrl: string | null;
}

interface Props {
  recipient: RecipientOption;
  options: RecipientOption[];
  onSelect: (id: string | null) => void;
}

export function RecipientChip({ recipient, options, onSelect }: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`Sending to ${recipient.name} — change recipient`}
          className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium text-zinc-700 shadow-sm transition-colors hover:bg-neutral-50"
        >
          <RecipientAvatar option={recipient} />
          {recipient.name}
          <CaretDownIcon className="size-3 text-zinc-400" weight="bold" />
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
