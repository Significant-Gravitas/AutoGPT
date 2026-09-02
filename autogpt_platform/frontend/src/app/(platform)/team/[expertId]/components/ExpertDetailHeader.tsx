"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";

interface Props {
  expert: Expert;
}

export function ExpertDetailHeader({ expert }: Props) {
  const accent = getRaisedExpertAccent(expert.role, expert.color);

  return (
    <header className="flex flex-col gap-4 sm:flex-row sm:items-center">
      <Avatar className="h-16 w-16 bg-white shadow-sm ring-1 ring-black/5">
        {expert.avatar_url ? (
          <AvatarImage src={expert.avatar_url} alt={expert.name} />
        ) : null}
        <AvatarFallback>{expert.name}</AvatarFallback>
      </Avatar>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
            {expert.name}
          </h1>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
              accent.pill,
            )}
          >
            <Icon icon={accent.roleIcon} size={14} />
            {expert.role}
          </span>
        </div>
        {expert.tagline ? (
          <p className="mt-1.5 text-base text-zinc-500">{expert.tagline}</p>
        ) : null}
      </div>
      <Button
        as="NextLink"
        href={`/copilot?expertId=${expert.id}`}
        variant="primary"
        size="small"
        className="shrink-0"
      >
        Chat
      </Button>
    </header>
  );
}
