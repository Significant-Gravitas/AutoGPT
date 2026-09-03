import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";

interface Props {
  expert: Expert;
  accent: ExpertAccent;
}

export function ExpertPageHeader({ expert, accent }: Props) {
  return (
    <header
      className={cn(
        "relative flex flex-col gap-6 overflow-hidden rounded-3xl border border-zinc-200/60 p-6 sm:flex-row sm:items-center sm:p-8",
        accent.washWide,
      )}
    >
      <Avatar className="h-28 w-28 bg-white shadow-sm ring-1 ring-black/5">
        {expert.avatar_url ? (
          <AvatarImage src={expert.avatar_url} alt={expert.name} />
        ) : null}
        <AvatarFallback>{expert.name}</AvatarFallback>
      </Avatar>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-4xl font-semibold tracking-[-0.02em] text-zinc-900">
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
          <p className="mt-2 max-w-2xl text-lg text-zinc-600">
            {expert.tagline}
          </p>
        ) : null}
      </div>
    </header>
  );
}
