import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
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

import { SpendMeter } from "./components/SpendMeter";
import {
  getNeedsSetupCount,
  getScheduleCountLabel,
  getWeeklySpend,
  ROW_CLASS,
  ROW_LINK_CLASS,
} from "../../helpers";
import { FireExpertDialog } from "../FireExpertDialog/FireExpertDialog";
import { FireExpertMenu } from "../FireExpertMenu/FireExpertMenu";
import { useExpertRow } from "./useExpertRow";

const ICON_ACTION_CLASS =
  "border-transparent text-zinc-500 hover:border-zinc-200 hover:bg-zinc-100 hover:text-zinc-900";

/** Actions stay out of the way on pointer devices and appear on hover or when
 *  something inside them takes focus. Below `lg` there is no hover to rely on,
 *  so they are always visible. */
const HOVER_ACTIONS_CLASS =
  "lg:opacity-0 lg:transition-opacity lg:group-hover:opacity-100 lg:group-focus-within:opacity-100";

interface Props {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
  pods: ExpertPod[];
  currentPod: ExpertPod | undefined;
  onInstallWorkflow: (expertId: string) => void;
  onEditSoul: (expertId: string) => void;
  onAssignPod: (expertId: string, podId: string | null) => void;
}

export function ExpertRow({
  expert,
  schedules,
  pods,
  currentPod,
  onInstallWorkflow,
  onEditSoul,
  onAssignPod,
}: Props) {
  const workflowCount = expert.workflows.length;
  const needsSetupCount = getNeedsSetupCount(expert, schedules);
  const scheduleLabel = getScheduleCountLabel(schedules);
  const weeklySpend = getWeeklySpend(expert);
  const { handleResume, isResuming, isFireOpen, openFire, closeFire } =
    useExpertRow(expert.id);
  const isPaused = Boolean(expert.schedules_paused_at);
  const tagline = expert.tagline || expert.identity;

  function handleInstallClick() {
    onInstallWorkflow(expert.id);
  }

  function handleEditSoulClick(event: MouseEvent) {
    event.stopPropagation();
    onEditSoul(expert.id);
  }

  return (
    <div className={`group ${ROW_CLASS} transition-colors hover:bg-zinc-50`}>
      <Link
        href={`/team/${expert.id}`}
        aria-label={`View ${expert.name}`}
        className={ROW_LINK_CLASS}
      />

      <ExpertAvatar
        name={expert.name}
        avatarUrl={expert.avatar_url ?? null}
        size={36}
        className="relative"
      />

      <div className="relative min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" className="truncate">
            {expert.name}
          </Text>
          {needsSetupCount > 0 ? (
            <Badge variant="warning" size="small" className="shrink-0">
              {needsSetupCount} {needsSetupCount === 1 ? "needs" : "need"} setup
            </Badge>
          ) : null}
          {isPaused ? (
            <Badge variant="warning" size="small" className="shrink-0">
              Schedules paused
            </Badge>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5">
          <Text variant="small" className="shrink-0 text-zinc-500">
            {expert.role}
          </Text>
          {tagline ? (
            <>
              <span aria-hidden className="text-zinc-300">
                ·
              </span>
              <Text variant="small" className="truncate text-zinc-500">
                {tagline}
              </Text>
            </>
          ) : null}
        </div>
      </div>

      <div className="relative hidden shrink-0 items-center gap-5 sm:flex">
        {weeklySpend ? (
          <div className="hidden w-28 flex-col gap-1 lg:flex">
            <Text
              variant="small"
              unmask={false}
              className="tabular-nums text-zinc-500"
            >
              {`${creditsToUsdLabel(weeklySpend.spent)} / ${creditsToUsdLabel(weeklySpend.budget)}`}
            </Text>
            <SpendMeter
              spent={weeklySpend.spent}
              budget={weeklySpend.budget}
              className="h-1"
            />
          </div>
        ) : null}
        <Text variant="small" className="text-zinc-500">
          {workflowCount} {workflowCount === 1 ? "workflow" : "workflows"}
        </Text>
        {scheduleLabel ? (
          <Text variant="small" className="text-zinc-500">
            {scheduleLabel}
          </Text>
        ) : null}
      </div>

      {isPaused ? (
        <Button
          variant="secondary"
          size="small"
          loading={isResuming}
          onClick={handleResume}
          className="relative shrink-0"
        >
          Resume schedules
        </Button>
      ) : null}

      <div
        className={`relative flex shrink-0 items-center gap-1 ${HOVER_ACTIONS_CLASS}`}
      >
        <Button
          as="NextLink"
          href={`/copilot?expertId=${expert.id}`}
          variant="ghost"
          size="small"
          className="hidden text-zinc-600 lg:inline-flex"
        >
          Chat
        </Button>
        <Button
          variant="icon"
          size="small"
          aria-label="Install workflow"
          className={ICON_ACTION_CLASS}
          onClick={handleInstallClick}
        >
          <Icon icon={PlusSignIcon} size={16} />
        </Button>
        <Button
          variant="icon"
          size="small"
          aria-label="Edit Soul"
          className={ICON_ACTION_CLASS}
          onClick={handleEditSoulClick}
        >
          <Icon icon={PencilEdit02Icon} size={16} />
        </Button>
        {pods.length > 0 ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="icon"
                size="small"
                className={ICON_ACTION_CLASS}
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
        <FireExpertMenu
          expertName={expert.name}
          onFire={openFire}
          testId="expert-row-actions"
          triggerClassName="text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
        />
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
