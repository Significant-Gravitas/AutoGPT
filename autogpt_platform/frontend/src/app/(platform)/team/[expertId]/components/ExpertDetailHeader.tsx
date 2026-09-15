"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useExpertPackageDownload } from "@/services/experts/useExpertPackageDownload";
import {
  BubbleChatIcon,
  Download01Icon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { getExpertCover } from "../../helpers";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { IntegrationIcons } from "../../components/ExpertTeamCard/components/IntegrationIcons";
import { ExpertAvatarButton } from "./ExpertAvatarButton/ExpertAvatarButton";

interface Props {
  expert: Expert;
  /** ``EXPERT_PORTABILITY``: without it the export route 404s, so the button
   *  would only ever produce an error toast. */
  canExport: boolean;
  onEditSoul: () => void;
  onChat: () => void;
}

export function ExpertDetailHeader({
  expert,
  canExport,
  onEditSoul,
  onChat,
}: Props) {
  const accent = getRaisedExpertAccent(expert.role, expert.color);
  const cover = getExpertCover(expert);
  const { isDownloading, download } = useExpertPackageDownload({
    kind: "expert",
    id: expert.id,
    name: expert.name,
    workflowCount: expert.workflows.length,
    skillCount: expert.skills.length,
  });

  return (
    <header>
      <ExpertCover className="h-36" color={cover.color} art={cover.art} />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <span className="-mt-12 ml-14 block shrink-0">
          <ExpertAvatarButton expert={expert} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
              {expert.name}
            </h1>
            {/* Same pill as the marketplace expert card. */}
            <Text
              variant="small-medium"
              as="span"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5",
                accent.pill,
              )}
            >
              <Icon icon={accent.roleIcon} size={12} />
              {expert.role}
            </Text>
            <IntegrationIcons
              expertName={expert.name}
              providers={expert.credential_providers ?? []}
            />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {/* Icon-only: the atom shows the aria-label as a hover tooltip. */}
          {canExport ? (
            <Button
              variant="icon"
              size="icon"
              aria-label="Download as file"
              loading={isDownloading}
              onClick={download}
              data-testid="expert-export-button"
            >
              <Icon icon={Download01Icon} size={16} />
            </Button>
          ) : null}
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
