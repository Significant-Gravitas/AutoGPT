"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, Tick02Icon } from "@hugeicons/core-free-icons";

interface Props {
  scopeExpertID: string | null;
  experts: Expert[];
  onSelect: (expertID: string | null) => void;
}

function AutoPilotAvatar({ size }: { size: number }) {
  return (
    <div
      style={{ width: size, height: size }}
      className="flex shrink-0 items-center justify-center rounded-full bg-violet-600 font-semibold text-white"
    >
      <span style={{ fontSize: size * 0.45 }}>A</span>
    </div>
  );
}

function ScopeCheck({ selected }: { selected: boolean }) {
  if (!selected) return null;
  return (
    <Icon
      icon={Tick02Icon}
      size={16}
      className="ml-auto shrink-0 text-violet-600"
    />
  );
}

export function ScopeSelect({ scopeExpertID, experts, onSelect }: Props) {
  const selectedExpert = scopeExpertID
    ? experts.find((expert) => expert.id === scopeExpertID)
    : undefined;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label="Memory scope"
          className={cn(
            "flex w-full items-center gap-2.5 rounded-[10px] border border-zinc-200 bg-white px-3 py-2 text-sm font-medium text-textBlack",
            "transition-colors hover:border-zinc-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400",
            "data-[state=open]:border-violet-500",
          )}
        >
          {selectedExpert ? (
            <ExpertAvatar
              name={selectedExpert.name}
              avatarUrl={selectedExpert.avatar_url}
              size={24}
            />
          ) : (
            <AutoPilotAvatar size={24} />
          )}
          <span className="min-w-0 truncate">
            {selectedExpert?.name ?? "AutoPilot"}
          </span>
          <Icon
            icon={ArrowDown01Icon}
            size={16}
            className="ml-auto shrink-0 text-zinc-500"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="max-h-96 w-[300px] overflow-y-auto rounded-[12px] p-1.5"
      >
        <DropdownMenuItem
          className="flex items-start gap-2.5 rounded-[8px] px-2.5 py-2"
          onSelect={() => onSelect(null)}
        >
          <AutoPilotAvatar size={28} />
          <span className="flex min-w-0 flex-col leading-snug">
            <span className="text-sm font-medium text-textBlack">
              AutoPilot
            </span>
            <span className="text-xs text-zinc-500">
              Your account memory — everything you do together
            </span>
          </span>
          <ScopeCheck selected={scopeExpertID === null} />
        </DropdownMenuItem>

        {experts.length > 0 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuLabel className="px-2.5 pb-0.5 pt-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-400">
              Your experts
            </DropdownMenuLabel>
            {experts.map((expert) => (
              <DropdownMenuItem
                key={expert.id}
                className="flex items-start gap-2.5 rounded-[8px] px-2.5 py-2"
                onSelect={() => onSelect(expert.id)}
              >
                <ExpertAvatar
                  name={expert.name}
                  avatarUrl={expert.avatar_url}
                  size={28}
                />
                <span className="flex min-w-0 flex-col leading-snug">
                  <span className="truncate text-sm font-medium text-textBlack">
                    {expert.name}
                  </span>
                  {expert.role && (
                    <span className="truncate text-xs text-zinc-500">
                      {expert.role}
                    </span>
                  )}
                </span>
                <ScopeCheck selected={scopeExpertID === expert.id} />
              </DropdownMenuItem>
            ))}
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
