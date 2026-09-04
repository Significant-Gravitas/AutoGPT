import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ReactNode } from "react";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";

interface Props {
  expert: Expert;
  accent: ExpertAccent;
  actions: ReactNode;
}

export function ExpertPageHeader({ expert, accent, actions }: Props) {
  return (
    <header className="flex flex-col gap-5 sm:flex-row sm:items-start">
      <div className="flex min-w-0 flex-1 items-start gap-4">
        <Avatar className="h-16 w-16 shrink-0 bg-white ring-1 ring-black/5">
          {expert.avatar_url ? (
            <AvatarImage src={expert.avatar_url} alt={expert.name} />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1 pt-0.5">
          <h1 className="text-2xl font-semibold leading-8 tracking-[-0.02em] text-zinc-900">
            {expert.name}
          </h1>
          <div className="mt-0.5 flex items-center gap-1.5 text-sm text-zinc-500">
            <Icon icon={accent.roleIcon} size={14} className={accent.icon} />
            {expert.role}
          </div>
          {expert.tagline ? (
            <p className="mt-3 max-w-prose text-[15px] leading-6 text-zinc-600">
              {expert.tagline}
            </p>
          ) : null}
        </div>
      </div>
      <div className="sm:shrink-0 sm:pt-1">{actions}</div>
    </header>
  );
}
