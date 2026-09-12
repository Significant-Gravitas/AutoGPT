import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertFace } from "@/app/(platform)/marketplace/components/ExpertsSection/components/ExpertFace";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";

interface Props {
  expert: Expert;
  accent: ExpertAccent;
  actions: ReactNode;
}

export function ExpertPageHeader({ expert, accent, actions }: Props) {
  return (
    <header>
      <div className="flex flex-wrap items-center gap-4 sm:gap-5">
        <ExpertFace expert={expert} size={72} className="h-18 w-18 shrink-0" />
        <div className="min-w-0 flex-1">
          <h1 className="text-[28px] font-semibold leading-8 tracking-[-0.02em] text-zinc-900">
            {expert.name}
          </h1>
          <span
            className={cn(
              "mt-2 inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium",
              accent.pill,
            )}
          >
            <Icon icon={accent.roleIcon} size={12} />
            {expert.role}
          </span>
        </div>
        <div className="w-full sm:w-auto">{actions}</div>
      </div>
      {expert.tagline ? (
        <p className="mt-5 max-w-[60ch] text-[17px] leading-7 text-zinc-600">
          {expert.tagline}
        </p>
      ) : null}
    </header>
  );
}
