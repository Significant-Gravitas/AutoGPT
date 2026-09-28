"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { CategoryTag } from "@/app/(platform)/marketplace/components/CategoryChip/CategoryTag";
import { BubbleChatIcon, PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { getExpertCover } from "../../helpers";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { IntegrationIcons } from "../../components/ExpertTeamCard/components/IntegrationIcons";
import { ExpertAvatarButton } from "./ExpertAvatarButton/ExpertAvatarButton";

interface Props {
  expert: Expert;
  onEditSoul: () => void;
  onChat: () => void;
}

export function ExpertDetailHeader({ expert, onEditSoul, onChat }: Props) {
  const cover = getExpertCover(expert);
  const topic = expert.categories?.[0];

  return (
    <header>
      <ExpertCover className="h-36" color={cover.color} art={cover.art} />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <span className="-mt-12 ml-14 block shrink-0">
          <ExpertAvatarButton expert={expert} />
        </span>
        <div className="min-w-0 flex-1">
          <ExpertIdentityDetails
            name={expert.name}
            role={expert.role}
            jobTitle={expert.job_title}
            size="page"
            nameAlign="baseline"
            nameAccessory={
              <IntegrationIcons
                expertName={expert.name}
                providers={expert.credential_providers ?? []}
              />
            }
          />
          {topic ? <CategoryTag category={topic} className="mt-2" /> : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="secondary"
            size="small"
            leadingIcon={PencilEdit02Icon}
            onClick={onEditSoul}
          >
            Edit Soul
          </Button>
          <Button
            variant="primary"
            size="small"
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
