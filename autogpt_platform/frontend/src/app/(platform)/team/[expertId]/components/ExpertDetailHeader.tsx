"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { getCategoryHex } from "@/components/molecules/ExpertAvatar/colors";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { CategoryTag } from "@/app/(platform)/marketplace/components/CategoryChip/CategoryTag";
import { BubbleChatIcon, PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { getExpertCover } from "../../helpers";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { IntegrationIcons } from "../../components/ExpertTeamCard/components/IntegrationIcons";
import { ExpertAvatarButton } from "./ExpertAvatarButton/ExpertAvatarButton";
import { ExpertBudgetSection } from "./ExpertBudgetSection";

interface Props {
  expert: Expert;
  onEditSoul: () => void;
  onChat: () => void;
}

export function ExpertDetailHeader({ expert, onEditSoul, onChat }: Props) {
  const cover = getExpertCover(expert);
  const topic = expert.categories?.[0];

  return (
    <header className="relative flex flex-col">
      <div className="relative">
        <ExpertCover className="h-36" color={cover.color} art={cover.art} />
        {topic ? (
          <CategoryTag
            category={topic}
            size="medium"
            className="absolute left-4 top-4 h-7 px-2.5 text-body-sm leading-inherit"
          />
        ) : null}
      </div>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <span className="relative z-10 -mt-12 ml-4 block shrink-0">
          <ExpertAvatarButton expert={expert} />
        </span>
        <div className="min-w-0 flex-1">
          <ExpertIdentityDetails
            name={expert.name}
            role={expert.role}
            jobTitle={expert.job_title}
            size="page"
            nameAlign="baseline"
            areaColor={getCategoryHex(topic)}
          />
          <div className="mt-2 flex items-center gap-2 empty:hidden">
            <IntegrationIcons
              expertName={expert.name}
              providers={expert.credential_providers ?? []}
            />
          </div>
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
      <ExpertBudgetSection
        expert={expert}
        color={cover.color}
        // TODO(tokens): no fitting role for ring-black/5 #141414; #141414 is foreground-base (a text colour)
        className="order-last mt-4 gap-2 rounded-xl bg-surface/75 px-3.5 py-2.5 shadow-sm ring-1 ring-black/5 backdrop-blur-md sm:absolute sm:right-4 sm:top-4 sm:order-none sm:mt-0 sm:w-64"
      />
    </header>
  );
}
