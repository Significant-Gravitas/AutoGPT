import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  PencilEdit02Icon,
  PlusSignIcon,
  Tick02Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import { creditsToUsdLabel } from "@/lib/credits";
import Link from "next/link";
import { MouseEvent } from "react";

import { ExpertCover } from "./components/ExpertCover";
import { SpendMeter } from "./components/SpendMeter";
import {
  getExpertBlurb,
  getExpertRosterStatus,
  getNeedsSetupCount,
  getWeeklySpend,
  ACTION_BUTTON_CLASS,
} from "../../helpers";
import { ChatCircle } from "@phosphor-icons/react";
import { CardStat, CardStats } from "../CardStats";
import { FireExpertDialog } from "../FireExpertDialog/FireExpertDialog";
import { FireExpertMenu } from "../FireExpertMenu/FireExpertMenu";
import { useExpertTeamCard } from "./useExpertTeamCard";

const COVER_ACTION_CLASS =
  "size-8 rounded-lg bg-white/90 p-0 text-zinc-700 backdrop-blur hover:bg-white hover:text-zinc-900";

/** The card's corners are heavily rounded, so its actions are too. */
const FOOTER_BUTTON_CLASS = `${ACTION_BUTTON_CLASS} flex-1`;
const FOOTER_OUTLINE_BUTTON_CLASS = `${FOOTER_BUTTON_CLASS} !border-zinc-200 hover:!border-zinc-300`;

interface Props {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
  pods: ExpertPod[];
  currentPod: ExpertPod | undefined;
  onInstallWorkflow: (expertId: string) => void;
  onEditSoul: (expertId: string) => void;
  onAssignPod: (expertId: string, podId: string | null) => void;
  onChat: (expertId: string) => void;
}

export function ExpertTeamCard({
  expert,
  schedules,
  pods,
  currentPod,
  onInstallWorkflow,
  onEditSoul,
  onAssignPod,
  onChat,
}: Props) {
  const blurb = getExpertBlurb(expert);
  const needsSetupCount = getNeedsSetupCount(expert, schedules);
  const rosterStatus = getExpertRosterStatus(expert, needsSetupCount);
  const weeklySpend = getWeeklySpend(expert);
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
    <div className="relative flex flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white">
      {/* Floated over the cover so the whole body stays one link target. */}
      <div className="absolute right-4 top-4 z-10 flex items-center gap-1.5">
        {pods.length > 0 ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="icon"
                size="small"
                className={COVER_ACTION_CLASS}
                aria-label={
                  currentPod
                    ? `Move to pod (currently ${currentPod.name})`
                    : "Move to pod"
                }
              >
                <Icon icon={UserGroupIcon} size={16} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              className="max-h-72 w-52 overflow-y-auto"
            >
              {pods.map((pod) => (
                <DropdownMenuItem
                  key={pod.id}
                  onSelect={() => onAssignPod(expert.id, pod.id)}
                >
                  <span className="flex-1 truncate">{pod.name}</span>
                  {expert.pod_id === pod.id ? (
                    <Icon icon={Tick02Icon} size={16} className="ml-2" />
                  ) : null}
                </DropdownMenuItem>
              ))}
              {expert.pod_id ? (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    onSelect={() => onAssignPod(expert.id, null)}
                  >
                    Remove from pod
                  </DropdownMenuItem>
                </>
              ) : null}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
        <Button
          variant="icon"
          size="small"
          aria-label="Edit Soul"
          className={COVER_ACTION_CLASS}
          onClick={handleEditSoulClick}
        >
          <Icon icon={PencilEdit02Icon} size={16} />
        </Button>
        <FireExpertMenu
          expertName={expert.name}
          onFire={openFire}
          testId="expert-card-actions"
          triggerClassName={COVER_ACTION_CLASS}
        />
      </div>

      <Link
        href={`/team/${expert.id}`}
        aria-label={`View ${expert.name}`}
        className="flex flex-1 flex-col items-center p-2 pb-4"
      >
        <ExpertCover color={expert.color} status={rosterStatus} />

        <div className="flex w-full items-start gap-3 px-2">
          <span className="-mt-11 ml-1 block shrink-0">
            <Avatar className="size-[5.25rem] rounded-full ring-4 ring-white">
              {expert.avatar_url ? (
                <AvatarImage
                  src={expert.avatar_url}
                  alt={expert.name}
                  width={84}
                  height={84}
                  className="bg-white"
                />
              ) : null}
              <AvatarFallback className="grain-overlay">
                {expert.name}
              </AvatarFallback>
            </Avatar>
          </span>

          <div className="mt-2 flex min-w-0 flex-1 flex-col gap-1">
            <div className="flex items-baseline justify-between gap-2">
              <Text variant="small-medium" className="text-zinc-700">
                Budget
              </Text>
              <Text
                variant="small-medium"
                unmask={false}
                className="tabular-nums text-zinc-700"
              >
                {weeklySpend
                  ? `${creditsToUsdLabel(weeklySpend.spent)} / ${creditsToUsdLabel(weeklySpend.budget)}`
                  : "No budget"}
              </Text>
            </div>
            <SpendMeter
              spent={weeklySpend?.spent ?? 0}
              budget={weeklySpend?.budget ?? 1}
              color={expert.color}
              muted={!weeklySpend}
            />
          </div>
        </div>

        <div className="mt-2 flex w-full flex-col items-start gap-1 px-2 pl-5 text-left">
          {/* `truncate` clips at the padding box, so descenders in a name like
              "Fiona Gray" need a little room below the line box. */}
          <Text variant="h4" className="w-full truncate pb-1">
            {expert.name}
          </Text>
          <Text variant="body" className="line-clamp-2 text-zinc-500">
            {expert.role}
          </Text>
          <Text variant="body" className="mt-1 line-clamp-2 text-zinc-500">
            {blurb}
          </Text>
          {needsSetupCount > 0 ? (
            <Badge variant="warning" size="small" className="mt-1">
              {needsSetupCount} {needsSetupCount === 1 ? "needs" : "need"} setup
            </Badge>
          ) : null}
        </div>

        <div className="w-full px-2">
          <CardStats className="mt-3 w-full">
            <CardStat label="Schedules">{schedules.length}</CardStat>
            <CardStat label="Skills">{expert.skills.length}</CardStat>
            <CardStat label="Workflows">{expert.workflows.length}</CardStat>
            <CardStat label="Integrations">
              {expert.credential_count ?? 0}
            </CardStat>
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
            className={ACTION_BUTTON_CLASS}
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
          className={FOOTER_BUTTON_CLASS}
          leftIcon={<ChatCircle size={14} />}
          onClick={() => onChat(expert.id)}
        >
          Chat
        </Button>
        <Button
          variant="outline"
          size="small"
          className={FOOTER_OUTLINE_BUTTON_CLASS}
          leftIcon={<Icon icon={PlusSignIcon} size={14} />}
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
