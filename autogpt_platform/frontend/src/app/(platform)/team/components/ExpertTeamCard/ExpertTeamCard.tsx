import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
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
import Image from "next/image";
import Link from "next/link";
import { MouseEvent } from "react";

import { CreditsMeter } from "./components/CreditsMeter";
import {
  getNeedsSetupCount,
  getScheduleCountLabel,
  getWeeklySpend,
} from "../../helpers";
import { FireExpertDialog } from "../FireExpertDialog/FireExpertDialog";
import { FireExpertMenu } from "../FireExpertMenu/FireExpertMenu";
import { useExpertTeamCard } from "./useExpertTeamCard";

const TEAM_CARD_BANNER_SRC = "/images/team-card-banner.jpg";

interface Props {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
  pods: ExpertPod[];
  currentPod: ExpertPod | undefined;
  onInstallWorkflow: (expertId: string) => void;
  onEditSoul: (expertId: string) => void;
  onAssignPod: (expertId: string, podId: string | null) => void;
}

export function ExpertTeamCard({
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
    <div className="flex flex-col rounded-[1.75rem] border border-zinc-200 bg-white p-1 transition-all duration-200 hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]">
      <div className="relative h-24 w-full overflow-hidden rounded-t-[1.5rem]">
        <Image
          src={TEAM_CARD_BANNER_SRC}
          alt=""
          fill
          sizes="(min-width: 1024px) 33vw, (min-width: 768px) 50vw, 100vw"
          className="object-cover"
        />
        {/* Fades the banner into the card background along its bottom edge, so
            the avatar straddling the seam has no hard line to cross. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 shadow-[inset_0_-4rem_3rem_-1.5rem_hsl(var(--card))]"
        />
        <div className="absolute right-3 top-3 z-10 flex items-center gap-1">
          <Button
            variant="icon"
            size="small"
            aria-label="Edit Soul"
            className="bg-white p-2 hover:bg-zinc-100"
            onClick={handleEditSoulClick}
          >
            <Icon icon={PencilEdit02Icon} size={16} />
          </Button>
          <FireExpertMenu
            expertName={expert.name}
            onFire={openFire}
            testId="expert-card-actions"
            triggerClassName="bg-white text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
          />
        </div>
      </div>
      <div className="flex flex-1 flex-col gap-3 p-4 pt-0">
        <Link
          href={`/team/${expert.id}`}
          aria-label={`View ${expert.name}`}
          className="flex flex-col gap-3"
        >
          <div className="flex items-center gap-3">
            {/* Half the avatar sits over the banner, half below it. */}
            <Avatar className="-mt-9 h-[4.5rem] w-[4.5rem] self-start ring-4 ring-white">
              {expert.avatar_url ? (
                <AvatarImage
                  src={expert.avatar_url}
                  alt={expert.name}
                  width={72}
                  height={72}
                  className="bg-white"
                />
              ) : null}
              <AvatarFallback>{expert.name}</AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <Text variant="large-medium">{expert.name}</Text>
              <Text variant="small" className="text-zinc-500">
                {expert.role}
              </Text>
            </div>
          </div>
          <Text variant="body" className="line-clamp-2 min-h-12 text-zinc-600">
            {expert.tagline || expert.identity}
          </Text>
          <div className="flex flex-col gap-1">
            <div className="flex items-baseline justify-between gap-2">
              <Text variant="small" className="text-zinc-500">
                Credits this week
              </Text>
              <Text
                variant="small"
                unmask={false}
                className="tabular-nums text-zinc-500"
              >
                {weeklySpend
                  ? `${weeklySpend.spent} / ${weeklySpend.budget}`
                  : "No budget"}
              </Text>
            </div>
            <CreditsMeter
              spent={weeklySpend?.spent ?? 0}
              budget={weeklySpend?.budget ?? 1}
              muted={!weeklySpend}
            />
          </div>
          <Text variant="small" className="min-h-5 text-zinc-500">
            {scheduleLabel ?? "No schedules yet"}
          </Text>
          <div className="flex min-h-5 items-center gap-2">
            <Text variant="small" className="text-zinc-500">
              {workflowCount} {workflowCount === 1 ? "workflow" : "workflows"}
            </Text>
            {needsSetupCount > 0 ? (
              <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 ring-1 ring-inset ring-amber-200">
                {needsSetupCount} {needsSetupCount === 1 ? "needs" : "need"}{" "}
                setup
              </span>
            ) : null}
          </div>
        </Link>
        {isPaused ? (
          <div className="flex items-center justify-between gap-2 rounded-xl bg-amber-50 px-3 py-2 ring-1 ring-inset ring-amber-200">
            <Text variant="small" className="text-amber-700">
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
        <div className="mt-auto flex flex-wrap gap-2">
          <Button
            as="NextLink"
            href={`/copilot?expertId=${expert.id}`}
            variant="secondary"
            size="small"
          >
            Chat
          </Button>
          <Button
            variant="ghost"
            size="small"
            leftIcon={<Icon icon={PlusSignIcon} size={16} />}
            onClick={handleInstallClick}
          >
            Install workflow
          </Button>
          {pods.length > 0 ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="small"
                  leftIcon={<Icon icon={UserGroupIcon} size={16} />}
                  aria-label={
                    currentPod
                      ? `Move to pod (currently ${currentPod.name})`
                      : "Move to pod"
                  }
                >
                  {currentPod ? currentPod.name : "Move to pod"}
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
        </div>
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
