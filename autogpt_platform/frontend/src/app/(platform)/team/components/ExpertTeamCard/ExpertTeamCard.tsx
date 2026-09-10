import { Expert } from "@/app/api/__generated__/models/expert";
import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  BubbleChatIcon,
  Calendar03Icon,
  FlashIcon,
  PencilEdit02Icon,
  PlusSignIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { MouseEvent } from "react";

import { ExpertCover } from "./components/ExpertCover";
import { IntegrationIcons } from "./components/IntegrationIcons";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  expertAvatarConfig,
  isUploadedAvatar,
} from "@/components/molecules/BotAvatar/helpers";

import { SpendMeter } from "./components/SpendMeter";
import {
  getExpertBlurb,
  getExpertCover,
  getExpertRosterStatus,
  getWeeklySpend,
} from "../../helpers";
import { CardStat, CardStats } from "../CardStats";
import { FireExpertDialog } from "../FireExpertDialog/FireExpertDialog";
import { FireExpertMenu } from "../FireExpertMenu/FireExpertMenu";
import { useExpertTeamCard } from "./useExpertTeamCard";

interface Props {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
  onInstallWorkflow: (expertId: string) => void;
  onEditSoul: (expertId: string) => void;
  onChat: (expertId: string) => void;
}

export function ExpertTeamCard({
  expert,
  schedules,
  onInstallWorkflow,
  onEditSoul,
  onChat,
}: Props) {
  const blurb = getExpertBlurb(expert);
  const accent = getRaisedExpertAccent(expert.role, expert.color);
  const rosterStatus = getExpertRosterStatus(expert);
  const weeklySpend = getWeeklySpend(expert);
  const cover = getExpertCover(expert);
  const { handleResume, isResuming, isFireOpen, openFire, closeFire } =
    useExpertTeamCard(expert.id);
  const isPaused = Boolean(expert.schedules_paused_at);

  function handleInstallClick() {
    onInstallWorkflow(expert.id);
  }

  function handleEditSoulClick(event: MouseEvent) {
    event.stopPropagation();
    onEditSoul(expert.id);
  }

  return (
    <div className="relative flex flex-col overflow-hidden rounded-2xl bg-white smooth-shadow-ring-sm">
      {/* Floated over the cover so the whole body stays one link target. */}
      <div className="absolute right-4 top-4 z-10 flex items-center gap-1.5">
        <Button
          variant="floating"
          size="icon-sm"
          aria-label="Edit Soul"
          leadingIcon={PencilEdit02Icon}
          onClick={handleEditSoulClick}
        />
        <FireExpertMenu
          expertName={expert.name}
          onFire={openFire}
          testId="expert-card-actions"
        />
      </div>

      <Link
        href={`/team/${expert.id}`}
        aria-label={`View ${expert.name}`}
        className="flex flex-1 flex-col items-center p-2 pb-4"
      >
        <ExpertCover
          color={cover.color}
          status={rosterStatus}
          art={cover.art}
        />

        <div className="flex w-full items-start gap-3 px-2">
          <span className="relative z-10 -mt-12 ml-1 block shrink-0">
            {isUploadedAvatar(expert.avatar_url) ? (
              <Avatar className="size-[5.5rem] rounded-full border border-stone-600 ring-4 ring-white">
                <AvatarImage
                  src={expert.avatar_url ?? undefined}
                  alt={expert.name}
                  width={88}
                  height={88}
                  className="bg-white"
                />
                <AvatarFallback className="grain-overlay">
                  {expert.name}
                </AvatarFallback>
              </Avatar>
            ) : (
              <span className="flex size-[5.5rem] items-center justify-center rounded-full border border-stone-600 bg-white ring-4 ring-white">
                <BotAvatar
                  config={expertAvatarConfig({
                    name: expert.name,
                    avatarUrl: expert.avatar_url,
                    color: expert.color,
                  })}
                  status="idle"
                  trackPointer
                  size={80}
                  title={expert.name}
                />
              </span>
            )}
          </span>

          <div className="mt-2 flex min-w-0 flex-1 flex-col gap-1">
            <div className="flex items-baseline justify-between gap-2">
              <Text variant="small-medium" tone="secondary">
                Budget
              </Text>
              <Text
                variant="small-medium"
                tone="secondary"
                unmask={false}
                className="tabular-nums"
              >
                {weeklySpend
                  ? `${creditsToUsdLabel(weeklySpend.spent)} / ${creditsToUsdLabel(weeklySpend.budget)}`
                  : "No budget"}
              </Text>
            </div>
            <SpendMeter
              spent={weeklySpend?.spent ?? 0}
              budget={weeklySpend?.budget ?? 1}
              muted={!weeklySpend}
            />
          </div>
        </div>

        <div className="mt-2 flex w-full flex-col items-start gap-1 px-2 pl-5 text-left">
          <div className="flex w-full items-center gap-2">
            {/* `truncate` clips at the padding box, so descenders in a name like
                "Fiona Gray" need a little room below the line box; the negative
                margin hands that room back so the logos centre on the text. */}
            <Text
              variant="lead-medium"
              tone="primary"
              className="-mb-1 min-w-0 truncate pb-1"
            >
              {expert.name}
            </Text>
            <IntegrationIcons
              expertName={expert.name}
              providers={expert.credential_providers ?? []}
            />
          </div>
          {/* Same pill as the expert page header and the marketplace card. */}
          <Text
            variant="small-medium"
            as="span"
            className={cn(
              "inline-flex max-w-full items-center gap-1.5 self-start rounded-full px-2.5 py-0.5",
              accent.pill,
            )}
          >
            <Icon icon={accent.roleIcon} size={12} className="shrink-0" />
            <span className="truncate">{expert.role}</span>
          </Text>
          <Text
            variant="body"
            tone="muted"
            className="mt-1 line-clamp-2 min-h-[2lh]"
          >
            {blurb}
          </Text>
        </div>

        <div className="mt-3 flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-2 pl-5">
          <CardStats>
            <CardStat
              icon={Calendar03Icon}
              label="Schedules"
              singular="schedule"
              count={schedules.length}
            />
            <CardStat
              icon={SparklesIcon}
              label="Skills"
              singular="skill"
              count={expert.skills.length}
            />
            <CardStat
              icon={FlashIcon}
              label="Workflows"
              singular="workflow"
              count={expert.workflows.length}
            />
          </CardStats>
        </div>
      </Link>

      {isPaused ? (
        <div className="mx-4 mb-3 flex items-center justify-between gap-2 rounded-lg bg-amber-50 px-3 py-2 ring-1 ring-inset ring-amber-200">
          <Text variant="body" className="text-amber-700">
            Schedules paused
          </Text>
          <Button
            variant="secondary"
            size="small"
            loading={isResuming}
            onClick={handleResume}
          >
            Resume schedules
          </Button>
        </div>
      ) : null}

      <div className="flex items-center gap-2 px-4 pb-4">
        <Button
          variant="secondary"
          size="small"
          className="flex-1"
          leadingIcon={BubbleChatIcon}
          onClick={() => onChat(expert.id)}
        >
          Chat
        </Button>
        <Button
          variant="secondary"
          size="small"
          className="flex-1"
          leadingIcon={PlusSignIcon}
          onClick={handleInstallClick}
        >
          Install workflow
        </Button>
      </div>

      <FireExpertDialog
        expertId={expert.id}
        expertName={expert.name}
        open={isFireOpen}
        onClose={closeFire}
      />
    </div>
  );
}
