"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { ExpertAvatarButton } from "./ExpertAvatarButton/ExpertAvatarButton";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";

interface Props {
  expert: Expert;
  onEditSoul: () => void;
}

export function ExpertDetailHeader({ expert, onEditSoul }: Props) {
  const accent = getRaisedExpertAccent(expert.role, expert.color);

  return (
    <header>
      <ExpertCover className="h-36" color={expert.color} />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <span className="-mt-12 ml-14 block shrink-0">
          <ExpertAvatarButton expert={expert} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
              {expert.name}
            </h1>
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-0.5 text-sm font-medium",
                accent.pill,
              )}
            >
              <Icon icon={accent.roleIcon} size={14} />
              {expert.role}
            </span>
          </div>
          {expert.tagline ? (
            <p className="mt-1 text-sm text-zinc-500">{expert.tagline}</p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="secondary"
            size="small"
            className={ACTION_BUTTON_CLASS}
            leftIcon={<Icon icon={PencilEdit02Icon} size={14} />}
            onClick={onEditSoul}
          >
            Edit Soul
          </Button>
          <Button
            as="NextLink"
            href={`/copilot?expertId=${expert.id}`}
            variant="primary"
            size="small"
            className={ACTION_BUTTON_CLASS}
          >
            Chat
          </Button>
        </div>
      </div>
    </header>
  );
}
