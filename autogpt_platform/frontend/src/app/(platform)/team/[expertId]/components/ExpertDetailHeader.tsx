"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { BubbleChatIcon, PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { ExpertAvatarButton } from "./ExpertAvatarButton/ExpertAvatarButton";

interface Props {
  expert: Expert;
  onEditSoul: () => void;
  onChat: () => void;
}

export function ExpertDetailHeader({ expert, onEditSoul, onChat }: Props) {
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
            <Text
              variant="body-medium"
              as="span"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-0.5",
                accent.pill,
              )}
            >
              <Icon icon={accent.roleIcon} size={14} />
              {expert.role}
            </Text>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="secondary"
            size="xs"
            leadingIcon={PencilEdit02Icon}
            onClick={onEditSoul}
          >
            Edit Soul
          </Button>
          <Button
            variant="primary"
            size="xs"
            leadingIcon={BubbleChatIcon}
            onClick={onChat}
          >
            Chat
          </Button>
        </div>
      </div>
    </header>
  );
}
