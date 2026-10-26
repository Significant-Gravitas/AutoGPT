"use client";

import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { DelegationExpert } from "../../../../../delegations";
import type { DelegationStatusView } from "../../../../../delegationViews";
import { BADGE_VARIANT } from "../helpers";

interface Props {
  expert: DelegationExpert;
  title: string;
  subtitle: string | null;
  view: DelegationStatusView;
  onBack: () => void;
}

export function DetailHeader({ expert, title, subtitle, view, onBack }: Props) {
  return (
    <>
      <button
        type="button"
        onClick={onBack}
        className="flex w-fit items-center gap-1 text-xs text-zinc-600 hover:text-zinc-900"
      >
        <Icon icon={ArrowLeft01Icon} size={14} />
        All work
      </button>
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex shrink-0 items-center gap-1">
            <AutopilotAvatar size={24} />
            <Icon icon={ArrowRight01Icon} size={12} className="text-zinc-400" />
            <ExpertAvatar
              name={expert.name}
              avatarUrl={expert.avatarUrl}
              color={expert.color}
              size={24}
            />
          </span>
          <span className="flex min-w-0 flex-col">
            <Text variant="h5" className="line-clamp-2">
              {title}
            </Text>
            {subtitle && (
              <span className="text-xs text-zinc-500">{subtitle}</span>
            )}
          </span>
        </div>
        <Badge variant={BADGE_VARIANT[view.tone]} className="shrink-0">
          {view.label}
        </Badge>
      </div>
    </>
  );
}
